# Import necessary libraries
import pandas as pd
import numpy as np

# Load the training and validation data
training_data = pd.read_csv('Training200.csv')
validation_data = pd.read_csv('Validation65.csv')

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in training_data.columns:
    training_data = training_data.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in validation_data.columns:
    validation_data = validation_data.drop(columns=['Unnamed: 0'])

# Create the target variable for the training data
NextDayUp_train = []
for i in range(len(training_data) - 1):
    if training_data.loc[i + 1, 'Open'] > training_data.loc[i, 'Close']:
        NextDayUp_train.append(1)  # Next day's opening price is greater
    else:
        NextDayUp_train.append(0)  # Not greater
# For the last day, the answer is 'Yes' as per your note
NextDayUp_train.append(1)
training_data['NextDayUp'] = NextDayUp_train

# Create the target variable for the validation data
NextDayUp_valid = []
for i in range(len(validation_data) - 1):
    if validation_data.loc[i + 1, 'Open'] > validation_data.loc[i, 'Close']:
        NextDayUp_valid.append(1)
    else:
        NextDayUp_valid.append(0)
# For the last day, assign '1' as we cannot compute the target
NextDayUp_valid.append(1)
validation_data['NextDayUp'] = NextDayUp_valid

# Remove any missing values
training_data.dropna(inplace=True)
validation_data.dropna(inplace=True)

# Convert 'Dividends' and 'Stock Splits' to strings if they exist
for col in ['Dividends', 'Stock Splits']:
    if col in training_data.columns:
        training_data[col] = training_data[col].astype(str)
    if col in validation_data.columns:
        validation_data[col] = validation_data[col].astype(str)

# Implement the PRISM algorithm
def prism_algorithm(data, target_class):
    rules = []
    data_remaining = data.copy()
    while len(data_remaining[data_remaining['NextDayUp'] == target_class]) > 0:
        rule_conditions = []
        data_subset = data_remaining.copy()
        while True:
            best_condition = None
            best_accuracy = 0
            best_subset = None
            features = data_subset.columns.drop(['NextDayUp'])
            for feature in features:
                unique_values = data_subset[feature].unique()
                if data_subset[feature].dtype == 'object':
                    # Categorical attribute
                    for value in unique_values:
                        condition = (data_subset[feature] == value)
                        subset = data_subset[condition]
                        if len(subset) == 0:
                            continue
                        accuracy = subset['NextDayUp'].mean()
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = (feature, '==', value)
                            best_subset = subset
                else:
                    # Continuous attribute
                    sorted_values = np.unique(data_subset[feature])
                    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
                    for threshold in thresholds:
                        # Condition: feature <= threshold
                        condition = (data_subset[feature] <= threshold)
                        subset = data_subset[condition]
                        if len(subset) == 0:
                            continue
                        accuracy = subset['NextDayUp'].mean()
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = (feature, '<=', threshold)
                            best_subset = subset
                        # Condition: feature > threshold
                        condition = (data_subset[feature] > threshold)
                        subset = data_subset[condition]
                        if len(subset) == 0:
                            continue
                        accuracy = subset['NextDayUp'].mean()
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = (feature, '>', threshold)
                            best_subset = subset
            if best_condition is None:
                break  # No further improvement
            rule_conditions.append(best_condition)
            data_subset = best_subset
            if best_accuracy == 1.0:
                break  # Perfect rule
        # Store the rule
        rule = {'conditions': rule_conditions, 'accuracy': best_accuracy, 'coverage': len(data_subset)}
        rules.append(rule)
        # Remove covered instances
        condition = pd.Series([True] * data_remaining.shape[0], index=data_remaining.index)
        for feature, operator, value in rule_conditions:
            if operator == '==':
                condition &= (data_remaining[feature] == value)
            elif operator == '<=':
                condition &= (data_remaining[feature] <= value)
            elif operator == '>':
                condition &= (data_remaining[feature] > value)
        data_remaining = data_remaining[~condition]
    return rules

# Generate rules using the PRISM algorithm
target_class = 1  # We are generating rules for NextDayUp = 1
rules = prism_algorithm(training_data, target_class)

# Output the generated rules
print("Generated Rules:")
for idx, rule in enumerate(rules):
    conditions = ' AND '.join([
        f"{feat} {op} {val if isinstance(val, str) else round(val, 4)}"
        for feat, op, val in rule['conditions']
    ])
    print(f"Rule {idx+1}: IF {conditions} THEN NextDayUp = {target_class} "
          f"(Accuracy: {rule['accuracy']*100:.2f}%, Coverage: {rule['coverage']})")

# Function to apply rules to data
def apply_rules(data, rules, default_class=0):
    predictions = []
    for idx, row in data.iterrows():
        predicted = False
        for rule in rules:
            match = True
            for feature, operator, value in rule['conditions']:
                if operator == '==':
                    if row[feature] != value:
                        match = False
                        break
                elif operator == '<=':
                    if not row[feature] <= value:
                        match = False
                        break
                elif operator == '>':
                    if not row[feature] > value:
                        match = False
                        break
            if match:
                predictions.append(target_class)
                predicted = True
                break
        if not predicted:
            predictions.append(default_class)
    return predictions

# Apply the rules to the validation data
validation_data['PredictedNextDayUp'] = apply_rules(validation_data, rules)

# Evaluate the performance
def evaluate_performance(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print("\nValidation Performance:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

evaluate_performance(validation_data['NextDayUp'], validation_data['PredictedNextDayUp'])
