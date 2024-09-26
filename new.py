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
    data.at[len(data) - 1, 'Next_Day_Price_Movement'] = 'YES'
    return data

# Apply target creation to the training data
training_data = create_target_variable(training_data)

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

# Step 4: Rule Application - Function to apply refined rules to the dataset
def apply_rules(data, rules):
    def evaluate_row(row):
        for rule in rules:
            # Check if all conditions in the rule hold true for the given row
            if all(row[condition[0]] == condition[1] for condition in rule['conditions']):
                return rule['prediction']
        return 'NO'  # Default prediction if no rule matches

    return data.apply(evaluate_row, axis=1)

# Generate and refine rules from the training data
initial_rules = generate_rules(training_data)
refined_rules = refine_rules(initial_rules)

# Step 5: Evaluate Rules on Validation Data
validation_data['Predicted_Next_Day_Price_Movement'] = apply_rules(validation_data, refined_rules)
accuracy = (validation_data['Predicted_Next_Day_Price_Movement'] == validation_data['Next_Day_Price_Movement']).mean()
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the refined rules to a file
with open("refined_rules.txt", "w") as file:
    for rule in refined_rules:
        conditions = ' and '.join([f"{cond[0]} == {cond[1]!r}" for cond in rule['conditions']])
        file.write(f"If {conditions}, then Predict {rule['prediction']} (Accuracy: {rule['accuracy']:.2f}, Coverage: {rule['coverage']:.2f})\n")

print("Refined rules saved to 'refined_rules.txt'.")