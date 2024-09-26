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
NextDayUp = []

for i in range(len(training_data) - 1):
    if training_data.loc[i + 1, 'Open'] > training_data.loc[i, 'Close']:
        NextDayUp.append(1)  # Next day's opening price is greater
    else:
        NextDayUp.append(0)  # Not greater

# For the last day, the answer is 'Yes' as per your note
NextDayUp.append(1)

# Add the target variable to the training data
training_data['NextDayUp'] = NextDayUp

# Remove missing values if any
training_data.dropna(inplace=True)

# Ensure 'Dividends' and 'Stock Splits' are strings if they exist
for col in ['Dividends', 'Stock Splits']:
    if col in training_data.columns:
        training_data[col] = training_data[col].astype(str)

# Define the function to find the best condition
def find_best_condition(data, target_class):
    max_accuracy = 0
    best_condition = None
    best_subset = pd.DataFrame()  # Initialize as empty DataFrame
    features = data.columns.drop(['NextDayUp'])

    for feature in features:
        if data[feature].dtype == 'object':
            # Categorical features
            for value in data[feature].unique():
                condition = (data[feature] == value)
                subset = data[condition]
                if len(subset) == 0:
                    continue
                accuracy = subset['NextDayUp'].mean()
                if accuracy > max_accuracy or (accuracy == max_accuracy and len(subset) > len(best_subset)):
                    max_accuracy = accuracy
                    best_condition = (feature, '==', value)
                    best_subset = subset.copy()
        else:
            # Numeric features
            unique_values = data[feature].unique()
            unique_values.sort()
            # Use percentiles to limit thresholds
            percentiles = np.percentile(unique_values, np.arange(5, 100, 5))
            thresholds = np.unique(percentiles)
            for threshold in thresholds:
                # Condition: feature <= threshold
                condition = (data[feature] <= threshold)
                subset = data[condition]
                if len(subset) == 0:
                    continue
                accuracy = subset['NextDayUp'].mean()
                if accuracy > max_accuracy or (accuracy == max_accuracy and len(subset) > len(best_subset)):
                    max_accuracy = accuracy
                    best_condition = (feature, '<=', threshold)
                    best_subset = subset.copy()
                # Condition: feature > threshold
                condition = (data[feature] > threshold)
                subset = data[condition]
                if len(subset) == 0:
                    continue
                accuracy = subset['NextDayUp'].mean()
                if accuracy > max_accuracy or (accuracy == max_accuracy and len(subset) > len(best_subset)):
                    max_accuracy = accuracy
                    best_condition = (feature, '>', threshold)
                    best_subset = subset.copy()
    return best_condition, max_accuracy, best_subset

# Initialize variables
rules = []
target_class = 1  # We are generating rules for NextDayUp = 1
data = training_data.copy()

# PRISM Algorithm
while len(data[data['NextDayUp'] == target_class]) > 0:
    current_conditions = []
    remaining_data = data.copy()

    while True:
        best_condition, max_accuracy, best_subset = find_best_condition(remaining_data, target_class)

        if best_condition is None or max_accuracy == 0:
            break  # Cannot improve the rule further

        current_conditions.append(best_condition)
        remaining_data = best_subset.copy()

        # Check if the rule is perfect or cannot be improved further
        if max_accuracy == 1.0:
            break

    if len(current_conditions) > 0:
        # Store the rule
        rule = {
            'conditions': current_conditions.copy(),
            'accuracy': max_accuracy,
            'coverage': len(remaining_data)
        }
        rules.append(rule)
        # Remove covered instances
        condition = pd.Series([True] * data.shape[0], index=data.index)  # Corrected line
        for feature, operator, value in current_conditions:
            if operator == '==':
                condition &= (data[feature] == value)
            elif operator == '<=':
                condition &= (data[feature] <= value)
            elif operator == '>':
                condition &= (data[feature] > value)
        data = data[~condition]
    else:
        break  # No more rules can be generated

# Output the generated rules
print("Generated Rules:")
for idx, rule in enumerate(rules):
    conditions = ' AND '.join([f"{feat} {op} {round(val, 4) if isinstance(val, float) else val}" for feat, op, val in rule['conditions']])
    print(f"Rule {idx+1}: IF {conditions} THEN NextDayUp = {target_class} (Accuracy: {rule['accuracy']*100:.2f}%, Coverage: {rule['coverage']})")

# Prepare the validation data
# Create the target variable for validation data
ValidationNextDayUp = []

for i in range(len(validation_data) - 1):
    if validation_data.loc[i + 1, 'Open'] > validation_data.loc[i, 'Close']:
        ValidationNextDayUp.append(1)
    else:
        ValidationNextDayUp.append(0)

# The last day's target cannot be computed; assign NaN
ValidationNextDayUp.append(np.nan)
validation_data['NextDayUp'] = ValidationNextDayUp

# Remove the last row since we cannot compute the target for it
validation_data = validation_data.iloc[:-1]

# Ensure 'Dividends' and 'Stock Splits' are strings if they exist
for col in ['Dividends', 'Stock Splits']:
    if col in validation_data.columns:
        validation_data[col] = validation_data[col].astype(str)

# Apply the rules to the validation data
def apply_rules_to_row(row, rules):
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
            return target_class  # NextDayUp = 1
    return 0  # Default class

validation_data['PredictedNextDayUp'] = validation_data.apply(lambda row: apply_rules_to_row(row, rules), axis=1)

# Evaluate the performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = validation_data['NextDayUp']
y_pred = validation_data['PredictedNextDayUp']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nValidation Performance:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
