# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats

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

# Feature Engineering
# You can add any additional features if needed

# Normalize continuous features
from pandas.api.types import is_numeric_dtype

features_to_normalize = [col for col in training_data.columns if is_numeric_dtype(training_data[col]) and col != 'NextDayUp']
for feature in features_to_normalize:
    min_value = training_data[feature].min()
    max_value = training_data[feature].max()
    training_data[feature] = (training_data[feature] - min_value) / (max_value - min_value)
    validation_data[feature] = (validation_data[feature] - min_value) / (max_value - min_value)

# Implement the PRISM algorithm with statistical threshold selection
def prism_algorithm(data, target_class, min_coverage=3, max_conditions=3):
    rules = []
    data_remaining = data.copy()
    while len(data_remaining[data_remaining['NextDayUp'] == target_class]) >= min_coverage:
        rule_conditions = []
        data_subset = data_remaining.copy()
        while True:
            best_condition = None
            best_p_value = 1  # Initialize with maximum p-value
            best_subset = None
            features = data_subset.columns.drop(['NextDayUp'])
            for feature in features:
                if data_subset[feature].dtype == 'object':
                    # Categorical attribute
                    unique_values = data_subset[feature].unique()
                    for value in unique_values:
                        condition = (data_subset[feature] == value)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage:
                            continue
                        # Perform chi-squared test
                        contingency_table = pd.crosstab(subset['NextDayUp'], subset[feature])
                        if contingency_table.shape[0] < 2:
                            continue
                        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                        if p < best_p_value:
                            best_p_value = p
                            best_condition = (feature, '==', value)
                            best_subset = subset
                else:
                    # Continuous attribute
                    # Use t-test to find the threshold that best separates the classes
                    thresholds = np.percentile(data_subset[feature], [25, 50, 75])
                    for threshold in thresholds:
                        condition = (data_subset[feature] <= threshold)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage or len(subset) == len(data_subset):
                            continue
                        group1 = subset['NextDayUp']
                        group2 = data_subset[~condition]['NextDayUp']
                        if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                            continue
                        t_stat, p = stats.ttest_ind(group1, group2, equal_var=False)
                        p = np.abs(p)
                        if p < best_p_value:
                            best_p_value = p
                            best_condition = (feature, '<=', threshold)
                            best_subset = subset
                        # Test for the opposite condition
                        condition = (data_subset[feature] > threshold)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage or len(subset) == len(data_subset):
                            continue
                        group1 = subset['NextDayUp']
                        group2 = data_subset[~condition]['NextDayUp']
                        if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                            continue
                        t_stat, p = stats.ttest_ind(group1, group2, equal_var=False)
                        p = np.abs(p)
                        if p < best_p_value:
                            best_p_value = p
                            best_condition = (feature, '>', threshold)
                            best_subset = subset
            if best_condition is None or len(rule_conditions) >= max_conditions:
                break  # No further improvement or max conditions reached
            rule_conditions.append(best_condition)
            data_subset = best_subset
            if best_p_value <= 0.01:
                break  # Statistically significant rule
        if len(data_subset) >= min_coverage:
            # Store the rule
            rule = {'conditions': rule_conditions, 'p_value': best_p_value, 'coverage': len(data_subset)}
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
        else:
            break  # No more rules with sufficient coverage
    return rules

# Generate rules using the PRISM algorithm with statistical threshold selection
target_class = 1  # We are generating rules for NextDayUp = 1
rules = prism_algorithm(training_data, target_class, min_coverage=3, max_conditions=3)

# Output the generated rules
print("Generated Rules:")
if not rules:
    print("No rules generated.")
else:
    for idx, rule in enumerate(rules):
        conditions = ' AND '.join([
            f"{feat} {op} {round(val, 4) if isinstance(val, float) else val}"
            for feat, op, val in rule['conditions']
        ])
        print(f"Rule {idx+1}: IF {conditions} THEN NextDayUp = {target_class} "
              f"(p-value: {rule['p_value']:.4f}, Coverage: {rule['coverage']})")

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
    return accuracy

# Evaluate performance on validation data
accuracy = evaluate_performance(validation_data['NextDayUp'], validation_data['PredictedNextDayUp'])
