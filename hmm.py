import pandas as pd
import numpy as np

# Load the training and validation data
training_data = pd.read_csv("Training200.csv")
validation_data = pd.read_csv("Validation65.csv")

# Step 1: Data Preparation - Create the target variable for training data based on next day's opening price
def create_target_variable(data):
    data['Next_Day_Price_Movement'] = 'NO'
    for i in range(len(data) - 1):
        if data.loc[i + 1, 'Open'] > data.loc[i, 'Close']:
            data.at[i, 'Next_Day_Price_Movement'] = 'YES'
    data.at[len(data) - 1, 'Next_Day_Price_Movement'] = 'YES'  # For the last day, we assume 'YES'
    return data

# Apply target creation to the training data
training_data = create_target_variable(training_data)

# Step 1.1: Apply target creation to the validation data (for evaluation purposes)
validation_data = create_target_variable(validation_data)

# Step 2: Rule Generation - Automatically generate rules based on feature conditions
def generate_rules(data, target_column='Next_Day_Price_Movement', min_coverage=0.05, min_accuracy=0.6):
    rules = []
    total_instances = len(data)

    # Iterate over each feature to generate conditions for rules
    for feature in data.columns:
        if feature not in [target_column]:
            unique_values = data[feature].unique()
            
            # Create rules for each unique value in the feature
            for value in unique_values:
                rule_condition = (feature, value)
                
                # Filter the dataset based on the rule condition
                subset = data[data[feature] == value]
                coverage = len(subset) / total_instances
                
                # Calculate rule accuracy
                if coverage >= min_coverage:
                    accuracy = (subset[target_column] == 'YES').mean()
                    
                    if accuracy >= min_accuracy:
                        rules.append({
                            'conditions': [rule_condition],
                            'prediction': 'YES',
                            'accuracy': accuracy,
                            'coverage': coverage
                        })

                    # Check for NO prediction
                    accuracy_no = (subset[target_column] == 'NO').mean()
                    if accuracy_no >= min_accuracy:
                        rules.append({
                            'conditions': [rule_condition],
                            'prediction': 'NO',
                            'accuracy': accuracy_no,
                            'coverage': coverage
                        })
    return rules

# Step 3: Rule Refinement - Combine, merge, and refine rules to improve accuracy
def refine_rules(rules):
    refined_rules = []

    for rule in rules:
        # Add rules to refined list if they are not redundant
        if rule not in refined_rules:
            refined_rules.append(rule)

    # Sort rules by accuracy and coverage
    refined_rules = sorted(refined_rules, key=lambda x: (-x['accuracy'], -x['coverage']))
    return refined_rules

# Step 4: Rule Application - Function to apply refined rules to the dataset and calculate accuracy and coverage
def apply_rules(data, rules):
    results = []
    for rule in rules:
        # Apply rule to find which instances it covers
        condition_str = ' & '.join([f"{condition[0]} == {repr(condition[1])}" for condition in rule['conditions']])
        covered_instances = data.query(condition_str)
        total_covered = len(covered_instances)
        
        # Calculate accuracy of the rule on the covered instances
        if total_covered > 0:
            rule_accuracy = (covered_instances['Next_Day_Price_Movement'] == rule['prediction']).mean()
            rule_coverage = total_covered / len(data)
            results.append({
                'rule': condition_str,
                'prediction': rule['prediction'],
                'accuracy': rule_accuracy,
                'coverage': rule_coverage,
                'total_covered': total_covered
            })
        else:
            results.append({
                'rule': condition_str,
                'prediction': rule['prediction'],
                'accuracy': 0,
                'coverage': 0,
                'total_covered': total_covered
            })
    return results

# Generate and refine rules from the training data
initial_rules = generate_rules(training_data)
refined_rules = refine_rules(initial_rules)

# Step 5: Evaluate Rules on Validation Data
validation_results = apply_rules(validation_data, refined_rules)

# Display each rule's performance
for result in validation_results:
    print(f"Rule: If {result['rule']} then Predict {result['prediction']}")
    print(f" - Accuracy: {result['accuracy']:.2f}, Coverage: {result['coverage']:.2f}, Total Covered: {result['total_covered']}")
    print("")

# Save the refined rules to a file
with open("refined_rules.txt", "w") as file:
    for result in validation_results:
        file.write(f"Rule: If {result['rule']} then Predict {result['prediction']} (Accuracy: {result['accuracy']:.2f}, Coverage: {result['coverage']:.2f}, Total Covered: {result['total_covered']})\n")

print("Refined rules with accuracy and coverage saved to 'refined_rules.txt'.")

# Function to load and apply rules to a new dataset
def load_and_apply_rules(input_csv, rules_file):
    # Load the new dataset
    new_data = pd.read_csv(input_csv)
    
    # Load the rules
    rules = []
    with open(rules_file, 'r') as file:
        for line in file:
            parts = line.split(", then Predict ")
            condition_part = parts[0].split("If ")[1]
            conditions = [(c.split(" == ")[0], eval(c.split(" == ")[1])) for c in condition_part.split(" and ")]
            prediction = parts[1].split(" ")[0]
            rules.append({'conditions': conditions, 'prediction': prediction})
    
    # Apply the rules to the new dataset
    new_data['Predicted_Next_Day_Price_Movement'] = apply_rules(new_data, rules)
    new_data.to_csv("output_predictions.csv", index=False)
    print("Predictions saved to 'output_predictions.csv'.")

# Example usage of the load_and_apply_rules function:
# load_and_apply_rules("Validation65.csv", "refined_rules.txt")
