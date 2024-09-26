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

# Convert 'Dividends' and 'Stock Splits' to categorical numerical codes
for col in ['Dividends', 'Stock Splits']:
    if col in training_data.columns:
        training_data[col] = training_data[col].astype('category').cat.codes
    if col in validation_data.columns:
        validation_data[col] = validation_data[col].astype('category').cat.codes

# Step 2: Feature Engineering

# 2.1 Lag Features
lag_features = ['Open', 'High', 'Low', 'Close', 'Volume']
for feature in lag_features:
    training_data[f'{feature}_lag1'] = training_data[feature].shift(1)
    training_data[f'{feature}_lag2'] = training_data[feature].shift(2)
    validation_data[f'{feature}_lag1'] = validation_data[feature].shift(1)
    validation_data[f'{feature}_lag2'] = validation_data[feature].shift(2)

# 2.2 Technical Indicators
# Moving Averages
training_data['MA_5'] = training_data['Close'].rolling(window=5).mean()
training_data['MA_10'] = training_data['Close'].rolling(window=10).mean()
validation_data['MA_5'] = validation_data['Close'].rolling(window=5).mean()
validation_data['MA_10'] = validation_data['Close'].rolling(window=10).mean()

# Exponential Moving Averages
training_data['EMA_5'] = training_data['Close'].ewm(span=5, adjust=False).mean()
training_data['EMA_10'] = training_data['Close'].ewm(span=10, adjust=False).mean()
validation_data['EMA_5'] = validation_data['Close'].ewm(span=5, adjust=False).mean()
validation_data['EMA_10'] = validation_data['Close'].ewm(span=10, adjust=False).mean()

# Relative Strength Index (RSI) already provided as RSI_05, RSI_09, RSI_14

# 2.3 Interaction Features
training_data['Volume_Price'] = training_data['Volume'] * training_data['Close']
validation_data['Volume_Price'] = validation_data['Volume'] * validation_data['Close']

# Remove NaN values created by lagging and rolling
training_data.dropna(inplace=True)
validation_data.dropna(inplace=True)

# Step 3: Data Preprocessing

# Normalize continuous features using Z-score normalization
from pandas.api.types import is_numeric_dtype

features_to_normalize = [col for col in training_data.columns if is_numeric_dtype(training_data[col]) and col != 'NextDayUp']
for feature in features_to_normalize:
    mean = training_data[feature].mean()
    std = training_data[feature].std()
    training_data[feature] = (training_data[feature] - mean) / std
    validation_data[feature] = (validation_data[feature] - mean) / std

# Step 4: Implementing the PRISM Algorithm

def calculate_entropy(data, target_attribute):
    """
    Calculate the entropy of the target attribute in the given data.
    """
    values, counts = np.unique(data[target_attribute], return_counts=True)
    entropy = 0
    for i in range(len(values)):
        p = counts[i] / np.sum(counts)
        entropy -= p * np.log2(p)
    return entropy

def calculate_information_gain(parent_set, subset, target_attribute):
    """
    Calculate the information gain of a subset with respect to the parent set.
    """
    parent_entropy = calculate_entropy(parent_set, target_attribute)
    subset_entropy = calculate_entropy(subset, target_attribute)
    weight = len(subset) / len(parent_set)
    info_gain = parent_entropy - weight * subset_entropy
    return info_gain

def find_best_condition(data, target_class, significance_level=0.05):
    """
    Find the best condition that maximizes information gain and is statistically significant.
    """
    best_info_gain = -np.inf
    best_condition = None
    best_subset = pd.DataFrame()
    
    features = data.columns.drop(['NextDayUp'])
    
    for feature in features:
        if data[feature].dtype == 'object' or data[feature].dtype.name == 'category':
            # Categorical feature
            unique_values = data[feature].unique()
            for value in unique_values:
                condition = (data[feature] == value)
                subset = data[condition]
                if len(subset) < 5:
                    continue
                # Chi-squared test
                contingency_table = pd.crosstab(subset['NextDayUp'], subset[feature])
                if contingency_table.shape[0] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(contingency_table)
                if p < significance_level and p < best_info_gain:
                    info_gain = calculate_information_gain(data, subset, 'NextDayUp')
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_condition = (feature, '==', value)
                        best_subset = subset
        else:
            # Continuous feature
            sorted_values = np.unique(data[feature])
            if len(sorted_values) <=1:
                continue
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
            for threshold in thresholds:
                # Condition: feature <= threshold
                condition = (data[feature] <= threshold)
                subset = data[condition]
                if len(subset) < 5:
                    continue
                group1 = subset['NextDayUp']
                group2 = data[~condition]['NextDayUp']
                if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                    continue
                t_stat, p = stats.ttest_ind(group1, group2, equal_var=False)
                p = np.abs(p)
                if p < significance_level and p < best_info_gain:
                    info_gain = calculate_information_gain(data, subset, 'NextDayUp')
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_condition = (feature, '<=', threshold)
                        best_subset = subset
                # Condition: feature > threshold
                condition = (data[feature] > threshold)
                subset = data[condition]
                if len(subset) < 5:
                    continue
                group1 = subset['NextDayUp']
                group2 = data[~condition]['NextDayUp']
                if len(np.unique(group1)) < 2 or len(np.unique(group2)) < 2:
                    continue
                t_stat, p = stats.ttest_ind(group1, group2, equal_var=False)
                p = np.abs(p)
                if p < significance_level and p < best_info_gain:
                    info_gain = calculate_information_gain(data, subset, 'NextDayUp')
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_condition = (feature, '>', threshold)
                        best_subset = subset
    return best_condition, best_info_gain, best_subset

def prism_algorithm(data, target_class, min_coverage=5, max_conditions=3, significance_level=0.05):
    """
    Implement the PRISM algorithm to generate classification rules.
    """
    rules = []
    data_remaining = data.copy()
    
    while len(data_remaining[data_remaining['NextDayUp'] == target_class]) >= min_coverage:
        rule_conditions = []
        data_subset = data_remaining.copy()
        while len(rule_conditions) < max_conditions:
            best_condition, best_info_gain, best_subset = find_best_condition(data_subset, target_class, significance_level)
            if best_condition is None or best_info_gain <=0:
                break
            rule_conditions.append(best_condition)
            data_subset = best_subset
            if best_info_gain == np.inf:
                break
        if len(rule_conditions) ==0:
            break
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
    return rules

# Step 5: Generate Rules Using PRISM Algorithm

# Define the target class
target_class = 1  # NextDayUp = 1 means the opening price of the next day is greater than the closing price of the current day

# Generate rules
rules = prism_algorithm(
    training_data,
    target_class,
    min_coverage=5,        # Minimum number of instances a rule must cover
    max_conditions=3,      # Maximum number of conditions per rule
    significance_level=0.05  # Significance level for statistical tests
)

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
              f"(Coverage: {rule['coverage']})")

# Step 6: Apply Rules to Validation Data and Evaluate Performance

def apply_rules(data, rules, default_class=0):
    """
    Apply the generated rules to the dataset to make predictions.
    """
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

# Make predictions on validation data
validation_data['PredictedNextDayUp'] = apply_rules(validation_data, rules, default_class=0)

# Step 7: Evaluate Performance

def evaluate_performance(y_true, y_pred):
    """
    Evaluate the performance of the model using various metrics.
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) >0 else 0
    recall = tp / (tp + fn) if (tp + fn) >0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall)>0 else 0
    
    print("\nValidation Performance:")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    
    return accuracy

# Evaluate the model
accuracy = evaluate_performance(validation_data['NextDayUp'], validation_data['PredictedNextDayUp'])
