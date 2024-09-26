# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats

# Step 1: Data Loading and Preprocessing

# Load the training and validation data
training_data = pd.read_csv('Training200.csv')
validation_data = pd.read_csv('Validation65.csv')

# Drop the 'Unnamed: 0' column if it exists
training_data = training_data.loc[:, ~training_data.columns.str.contains('^Unnamed')]
validation_data = validation_data.loc[:, ~validation_data.columns.str.contains('^Unnamed')]

# Create the target variable for the training data
NextDayUp_train = []
for i in range(len(training_data) - 1):
    if training_data.iloc[i + 1]['Open'] > training_data.iloc[i]['Close']:
        NextDayUp_train.append(1)  # Next day's opening price is greater
    else:
        NextDayUp_train.append(0)  # Not greater
# For the last day, assign '1' as per your note
NextDayUp_train.append(1)
training_data['NextDayUp'] = NextDayUp_train

# Create the target variable for the validation data
NextDayUp_valid = []
for i in range(len(validation_data) - 1):
    if validation_data.iloc[i + 1]['Open'] > validation_data.iloc[i]['Close']:
        NextDayUp_valid.append(1)
    else:
        NextDayUp_valid.append(0)
# For the last day, assign '1'
NextDayUp_valid.append(1)
validation_data['NextDayUp'] = NextDayUp_valid

# Remove any missing values
training_data.dropna(inplace=True)
validation_data.dropna(inplace=True)

# Convert 'Dividends' and 'Stock Splits' to categorical numeric codes
for col in ['Dividends', 'Stock Splits']:
    if col in training_data.columns:
        training_data[col] = training_data[col].astype('category').cat.codes
    if col in validation_data.columns:
        validation_data[col] = validation_data[col].astype('category').cat.codes

# Data Preprocessing: Normalize continuous features
from pandas.api.types import is_numeric_dtype

features_to_normalize = [col for col in training_data.columns if is_numeric_dtype(training_data[col]) and col != 'NextDayUp']
for feature in features_to_normalize:
    min_value = training_data[feature].min()
    max_value = training_data[feature].max()
    training_data[feature] = (training_data[feature] - min_value) / (max_value - min_value)
    validation_data[feature] = (validation_data[feature] - min_value) / (max_value - min_value)

# Step 2: Implement the Optimized PRISM Algorithm

def prism_algorithm(data, target_class, min_coverage=1, max_conditions=5, significance_level=0.5):
    rules = []
    data_remaining = data.copy()
    while len(data_remaining[data_remaining['NextDayUp'] == target_class]) >= min_coverage:
        rule_conditions = []
        data_subset = data_remaining.copy()
        while True:
            best_condition = None
            best_p_value = 1.0  # Initialize with maximum p-value
            best_subset = pd.DataFrame()
            features = data_subset.columns.drop(['NextDayUp'])
            for feature in features:
                if data_subset[feature].dtype == 'object' or data_subset[feature].dtype.name == 'category':
                    # Categorical attribute
                    unique_values = data_subset[feature].unique()
                    for value in unique_values:
                        condition = (data_subset[feature] == value)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage:
                            continue
                        # Chi-squared test
                        contingency_table = pd.crosstab(subset['NextDayUp'], subset[feature])
                        if contingency_table.shape[0] < 2:
                            continue
                        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_condition = (feature, '==', value)
                            best_subset = subset
                else:
                    # Continuous attribute
                    # Use more percentile points for thresholds
                    percentiles = np.percentile(data_subset[feature], np.linspace(5, 95, 19))
                    thresholds = np.unique(percentiles)
                    for threshold in thresholds:
                        # Condition: feature <= threshold
                        condition = (data_subset[feature] <= threshold)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage or len(subset) == len(data_subset):
                            continue
                        group1 = subset['NextDayUp']
                        group2 = data_subset[~condition]['NextDayUp']
                        if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                            continue
                        # T-test
                        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        p_value = np.abs(p_value)
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_condition = (feature, '<=', threshold)
                            best_subset = subset
                        # Condition: feature > threshold
                        condition = (data_subset[feature] > threshold)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage or len(subset) == len(data_subset):
                            continue
                        group1 = subset['NextDayUp']
                        group2 = data_subset[~condition]['NextDayUp']
                        if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                            continue
                        # T-test
                        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        p_value = np.abs(p_value)
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_condition = (feature, '>', threshold)
                            best_subset = subset
            if best_condition is None or len(rule_conditions) >= max_conditions:
                break  # No further improvement or maximum conditions reached
            if best_p_value > significance_level:
                break  # Condition not statistically significant
            rule_conditions.append(best_condition)
            data_subset = best_subset
        if len(data_subset) >= min_coverage and len(rule_conditions) > 0:
            # Store the rule
            rule = {'conditions': rule_conditions.copy(), 'coverage': len(data_subset)}
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
            break  # No more rules can be generated
    return rules

# Generate rules using the optimized PRISM algorithm
target_class = 1  # We are generating rules for NextDayUp = 1
rules = prism_algorithm(
    training_data,
    target_class,
    min_coverage=1,         # Allowing specific rules
    max_conditions=5,       # Allowing more complex rules
    significance_level=0.5  # Including more conditions
)

# Output the generated rules
print("Generated Rules:")
if not rules:
    print("No rules generated.")
else:
    for idx, rule in enumerate(rules):
        conditions = ' AND '.join([
            f"{feat} {op} {val if isinstance(val, str) else round(val, 4)}"
            for feat, op, val in rule['conditions']
        ])
        print(f"Rule {idx+1}: IF {conditions} THEN NextDayUp = {target_class} "
              f"(Coverage: {rule['coverage']})")

# Step 3: Apply the Rules and Evaluate Performance

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
