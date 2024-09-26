import pandas as pd
import numpy as np

# Load the data (assuming you have already read the CSVs)
training_data = pd.read_csv('Training200.csv')
training_data['Next_Open'] = training_data['Open'].shift(-1)
training_data['Target'] = (training_data['Next_Open'] > training_data['Close']).astype(int)
training_data.loc[199, 'Target'] = 1
training_data_processed = training_data.iloc[:-1].copy()

# Exclude 'Unnamed: 0' from the features
features = [col for col in training_data_processed.columns if col not in ['Unnamed: 0', 'Target', 'Next_Open']]

def calculate_coverage_accuracy(subset, target_column, target_value):
    """Calculate the coverage and accuracy of the rule on the subset."""
    coverage = len(subset)
    if coverage == 0:
        return 0, 0
    accuracy = sum(subset[target_column] == target_value) / coverage
    return coverage, accuracy

def prism_algorithm_complete(training_data, target_column, target_value, features):
    """
    Implements a complete version of the Prism algorithm considering all features and values.

    Parameters:
    - training_data: DataFrame containing the training data with feature columns and target column.
    - target_column: The name of the column representing the target variable.
    - target_value: The target value we want to create rules for (1 for YES, 0 for NO).
    - features: List of feature columns to consider for rule generation.

    Returns:
    - rules: List of generated rules with coverage and accuracy.
    """
    rules = []
    remaining_data = training_data.copy()
    
    # Loop to create rules for the target value
    while not remaining_data.empty:
        rule = []  # Start with an empty rule
        current_data = remaining_data.copy()
        best_coverage = 0
        best_accuracy = 0
        best_condition = None

        # Iterate to add conditions to the rule
        while True:
            best_accuracy = 0
            best_coverage = 0
            best_condition = None

            # Check each condition
            for column in features:
                # Test adding each condition (feature > value)
                for value in current_data[column].unique():
                    # Create conditions
                    condition = (column, '>', value)

                    # Apply condition to filter the data
                    subset = current_data[current_data[column] > value]

                    # Skip if the subset is empty
                    if len(subset) == 0:
                        continue

                    # Calculate the coverage and accuracy of the rule
                    coverage, accuracy = calculate_coverage_accuracy(subset, target_column, target_value)

                    # Choose the best condition based on accuracy and coverage
                    if accuracy > best_accuracy or (accuracy == best_accuracy and coverage > best_coverage):
                        best_accuracy = accuracy
                        best_coverage = coverage
                        best_condition = condition

            # Stop if no improvement
            if best_condition is None or best_accuracy <= 0.5:
                break

            # Add the best condition to the rule
            rule.append(best_condition)
            
            # Filter the data with the best condition
            column, operator, value = best_condition
            current_data = current_data[current_data[column] > value]

        # Stop if no complete rule is created
        if len(rule) == 0:
            break

        # Store the rule and remove the covered data
        rules.append((rule, best_coverage, best_accuracy))
        remaining_data = remaining_data[~remaining_data.index.isin(current_data.index)]

    return rules

# Generate rules for the target value 1 (YES) using all features and conditions
generated_rules_complete = prism_algorithm_complete(training_data_processed, 'Target', 1, features)

# Print generated rules with coverage and accuracy
for rule_idx, (rule, coverage, accuracy) in enumerate(generated_rules_complete, start=1):
    print(f"Rule {rule_idx} (Coverage: {coverage}, Accuracy: {accuracy}):")
    for condition in rule:
        print(f"  {condition[0]} {condition[1]} {condition[2]}")
