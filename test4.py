import pandas as pd
import numpy as np

# Load the data and prepare the target variable
def load_and_prepare_data(training_file):
    # Load the data
    data = pd.read_csv(training_file)
    
    # Create the target variable: Next day's opening price > today's closing price
    data['Next_Open'] = data['Open'].shift(-1)
    data['Target'] = (data['Next_Open'] > data['Close']).astype(int)
    
    # Assign 'YES' label for 200th day as given in the problem statement
    data.loc[199, 'Target'] = 1
    
    # Drop the last row as we cannot create a target for it
    data = data.iloc[:-1].copy()
    
    return data

# Function to calculate coverage and accuracy of a subset based on a target condition
def calculate_coverage_accuracy(subset, target_column, target_value):
    coverage = len(subset)
    if coverage == 0:
        return 0, 0
    accuracy = sum(subset[target_column] == target_value) / coverage
    return coverage, accuracy

# Function to mine rules for a given class (target_value) using the Prism algorithm
def mine_rules_for_class(training_data, target_column, target_value, features):
    rules = []
    remaining_data = training_data.copy()
    
    # Continue mining rules until no more data is left or new rules can be found
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

            # Evaluate conditions based on each feature and possible thresholds
            for column in features:
                if column in used_features:
                    continue  # Skip features already in the current rule
                
                # Determine potential thresholds (use quartiles or similar statistics)
                thresholds = current_data[column].quantile([0.25, 0.5, 0.75]).values

                # Test adding each condition (feature > threshold)
                for value in thresholds:
                    condition = (column, '>', value)
                    subset = current_data[current_data[column] > value]

                    # Skip empty subsets
                    if len(subset) == 0:
                        continue

                    # Calculate coverage and accuracy of the condition
                    coverage, accuracy = calculate_coverage_accuracy(subset, target_column, target_value)

                    # Keep track of the best condition based on accuracy and coverage
                    if accuracy > best_accuracy or (accuracy == best_accuracy and coverage > best_coverage):
                        best_accuracy = accuracy
                        best_coverage = coverage
                        best_condition = condition

            # Stop if no improvement or accuracy is too low
            if best_condition is None or best_accuracy <= 0.5:
                break

            # Add the best condition to the rule
            rule.append(best_condition)
            
            # Filter data with the best condition
            column, operator, value = best_condition
            current_data = current_data[current_data[column] > value]

        # Stop if no complete rule is created
        if len(rule) == 0:
            break

        # Store the rule and remove the covered data
        rules.append((rule, best_coverage, best_accuracy))
        remaining_data = remaining_data[~remaining_data.index.isin(current_data.index)]

    return rules

# Function to mine rules for all classes (YES and NO) in the dataset
def mine_rules(training_data, target_column, features):
    # Mine rules for the positive class (YES)
    yes_rules = mine_rules_for_class(training_data, target_column, 1, features)
    # Mine rules for the negative class (NO) if needed
    # no_rules = mine_rules_for_class(training_data, target_column, 0, features)
    # For simplicity, we will start with just the positive class rules
    return yes_rules

# Function to apply mined rules to a dataset and generate predictions
def apply_rules(rules, data):
    predictions = []
    for _, row in data.iterrows():
        prediction = 0  # Default prediction is 'NO'
        
        # Check each rule to see if it matches the row
        for rule, _, _ in rules:
            rule_matched = all(row[cond[0]] > cond[2] for cond in rule)
            if rule_matched:
                prediction = 1  # Set prediction to 'YES'
                break

        predictions.append(prediction)
    
    return predictions

# Main script to mine and validate rules
if __name__ == "__main__":
    # Load training data
    training_data = load_and_prepare_data('Training200.csv')
    
    # List of feature columns to use
    features = [col for col in training_data.columns if col not in ['Unnamed: 0', 'Target', 'Next_Open']]
    
    # Mine rules from training data
    mined_rules = mine_rules(training_data, 'Target', features)
    
    # Display mined rules
    print("\nMined Rules:")
    for rule_idx, (rule, coverage, accuracy) in enumerate(mined_rules, start=1):
        print(f"Rule {rule_idx} (Coverage: {coverage}, Accuracy: {accuracy}):")
        for condition in rule:
            print(f"  {condition[0]} {condition[1]} {condition[2]}")
    
    # Load validation data (if available) and apply mined rules
    validation_data = pd.read_csv('Validation65.csv')
    validation_predictions = apply_rules(mined_rules, validation_data)
    print(validation_predictions)