import pandas as pd
import numpy as np
from itertools import combinations

# Load and prepare the data
def load_and_prepare_data(training_file):
    data = pd.read_csv(training_file)
    data['Next_Open'] = data['Open'].shift(-1)
    data['Target'] = (data['Next_Open'] > data['Close']).astype(int)
    data.loc[199, 'Target'] = 1
    data = data.iloc[:-1].copy()
    return data

# Calculate coverage and accuracy
def calculate_coverage_accuracy(subset, target_column, target_value):
    coverage = len(subset)
    if coverage == 0:
        return 0, 0
    accuracy = sum(subset[target_column] == target_value) / coverage
    return coverage, accuracy

# Build rules based on association mining concepts
def generate_candidate_rules(training_data, target_column, features):
    rules = []
    for feature in features:
        # Use various statistical thresholds as potential rules
        thresholds = training_data[feature].quantile([0.25, 0.5, 0.75]).values
        for threshold in thresholds:
            rule = [(feature, '>', threshold)]
            coverage, accuracy = calculate_coverage_accuracy(
                training_data[training_data[feature] > threshold], target_column, 1
            )
            rules.append((rule, coverage, accuracy))
    return rules

# Refine rules by adding conditions and checking logical combinations
def refine_rules(training_data, target_column, rules, features, min_coverage=5, min_accuracy=0.6):
    refined_rules = []
    for base_rule, base_coverage, base_accuracy in rules:
        if base_coverage < min_coverage or base_accuracy < min_accuracy:
            continue  # Skip rules that don't meet basic criteria
        for feature in features:
            if any(cond[0] == feature for cond in base_rule):
                continue  # Skip if the feature is already in the rule

            # Test adding this feature with various thresholds
            thresholds = training_data[feature].quantile([0.25, 0.5, 0.75]).values
            for threshold in thresholds:
                new_rule = base_rule + [(feature, '>', threshold)]
                subset = training_data
                for cond in new_rule:
                    subset = subset[subset[cond[0]] > cond[2]]
                
                # Check for duplicate conditions and skip
                if len(set(new_rule)) < len(new_rule):
                    continue
                
                coverage, accuracy = calculate_coverage_accuracy(subset, target_column, 1)
                if coverage >= min_coverage and accuracy >= min_accuracy:
                    refined_rules.append((new_rule, coverage, accuracy))
    return refined_rules

# Combine rules using logical AND/OR operations and handle overlapping data correctly
def combine_rules(rules, training_data, target_column):
    combined_rules = []
    used_indices = set()
    for rule, coverage, accuracy in rules:
        subset = training_data.copy()
        for cond in rule:
            subset = subset[subset[cond[0]] > cond[2]]
        
        # Avoid double-counting by checking overlap with used indices
        rule_indices = set(subset.index)
        if rule_indices & used_indices:  # Check for overlap
            continue

        # Add to used indices to ensure no double counting
        used_indices.update(rule_indices)
        
        combined_rules.append((rule, len(rule_indices), accuracy))
    return combined_rules

# Function to handle exceptions in rules
def handle_exceptions(rules, training_data, target_column):
    exception_rules = []
    for rule, coverage, accuracy in rules:
        exception_subset = training_data
        for cond in rule:
            exception_subset = exception_subset[exception_subset[cond[0]] <= cond[2]]
        exception_coverage, exception_accuracy = calculate_coverage_accuracy(exception_subset, target_column, 1)
        if exception_coverage > 0:
            exception_rules.append((rule, exception_coverage, exception_accuracy, "Exception"))
    return exception_rules

# Apply rules to a dataset and generate predictions
def apply_rules(rules, data):
    predictions = []
    for _, row in data.iterrows():
        prediction = 0  # Default prediction is 'NO'
        for rule, _, _ in rules:
            if all(row[cond[0]] > cond[2] for cond in rule):
                prediction = 1  # Rule matches, predict 'YES'
                break
        predictions.append(prediction)
    return predictions

# Main script to mine, refine, and validate rules
if __name__ == "__main__":
    # Load training data
    training_data = load_and_prepare_data('Training200.csv')
    features = [col for col in training_data.columns if col not in ['Unnamed: 0', 'Target', 'Next_Open']]
    
    # Generate initial candidate rules
    candidate_rules = generate_candidate_rules(training_data, 'Target', features)
    
    # Refine the candidate rules to improve coverage and accuracy
    refined_rules = refine_rules(training_data, 'Target', candidate_rules, features)
    
    # Handle exceptions to the refined rules
    exception_rules = handle_exceptions(refined_rules, training_data, 'Target')
    
    # Combine rules using logical AND/OR for better generalization and handle overlap
    combined_rules = combine_rules(refined_rules, training_data, 'Target')
    
    # Display mined rules
    print("\nRefined and Combined Rules:")
    for rule_idx, (rule, coverage, accuracy) in enumerate(combined_rules, start=1):
        print(f"Rule {rule_idx} (Coverage: {coverage}, Accuracy: {accuracy}):")
        for condition in rule:
            print(f"  {condition[0]} {condition[1]} {condition[2]}")
            
    # Display exception rules
    print("\nException Rules:")
    for rule_idx, (rule, coverage, accuracy, label) in enumerate(exception_rules, start=1):
        print(f"Rule {rule_idx} (Coverage: {coverage}, Accuracy: {accuracy}) - {label}:")
        for condition in rule:
            print(f"  {condition[0]} {condition[1]} {condition[2]}")
