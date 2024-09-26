import pandas as pd
import numpy as np

# Load and prepare data
training_data = pd.read_csv('Training200.csv')
training_data['Next_Open'] = training_data['Open'].shift(-1)
training_data['Target'] = (training_data['Next_Open'] > training_data['Close']).astype(int)
training_data.loc[199, 'Target'] = 1
training_data_processed = training_data.iloc[:-1].copy()

# Exclude 'Unnamed: 0' and non-feature columns
features = [col for col in training_data_processed.columns if col not in ['Unnamed: 0', 'Target', 'Next_Open']]

def calculate_coverage_accuracy(subset, target_column, target_value):
    """Calculate the coverage and accuracy of the rule on the subset."""
    coverage = len(subset)
    if coverage == 0:
        return 0, 0
    accuracy = sum(subset[target_column] == target_value) / coverage
    return coverage, accuracy

def refined_prism_algorithm(training_data, target_column, target_value, features):
    """
    Implements a refined version of the Prism algorithm with better condition selection and redundancy control.

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
            used_features = [cond[0] for cond in rule]  # Features already used in this rule

            # Check each condition
            for column in features:
                if column in used_features:
                    continue  # Skip if this feature is already used in the current rule

                # Use predefined thresholds like quartiles
                thresholds = current_data[column].quantile([0.25, 0.5, 0.75]).values

                # Test adding each condition (feature > threshold)
                for value in thresholds:
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

# Generate rules for the target value 1 (YES) using refined Prism algorithm
try:
    refined_rules = refined_prism_algorithm(training_data_processed, 'Target', 1, features)
except Exception as e:
    print(f"Error in rule generation: {e}")

# Print generated rules with coverage and accuracy
print("\nGenerated Rules:")
for rule_idx, (rule, coverage, accuracy) in enumerate(refined_rules, start=1):
    print(f"Rule {rule_idx} (Coverage: {coverage}, Accuracy: {accuracy}):")
    for condition in rule:
        print(f"  {condition[0]} {condition[1]} {condition[2]}")
