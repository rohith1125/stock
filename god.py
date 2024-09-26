import pandas as pd
import numpy as np

# Load the training and validation data
training_data = pd.read_csv("Training200.csv")
validation_data = pd.read_csv("Validation65.csv")

# Create the target variable for training data based on next day's opening price
def create_target_variable(data):
    data['Next_Day_Price_Movement'] = 'NO'
    for i in range(len(data) - 1):
        if data.loc[i + 1, 'Open'] > data.loc[i, 'Close']:
            data.at[i, 'Next_Day_Price_Movement'] = 'YES'
    data.at[len(data) - 1, 'Next_Day_Price_Movement'] = 'YES'
    return data

# Apply target creation to the training data
training_data = create_target_variable(training_data)

# PRISM Algorithm Implementation
def prism(data, target_column='Next_Day_Price_Movement', target_class='YES', min_coverage=0.05):
    rules = []
    total_instances = len(data)

    while len(data) > 0:
        best_rule = {'conditions': [], 'accuracy': 0, 'coverage': 0}
        current_data = data.copy()

        while len(current_data) > 0:
            best_condition = None
            best_accuracy = 0
            best_coverage = 0

            # Evaluate all conditions based on the remaining data
            for feature in [col for col in data.columns if col not in [target_column, 'Next_Day_Price_Movement']]:
                unique_values = current_data[feature].unique()
                for value in unique_values:
                    condition = f"{feature} == {repr(value)}"  # Create condition as string
                    subset = current_data.query(condition)
                    coverage = len(subset) / total_instances

                    if coverage < min_coverage:
                        continue
                    
                    accuracy = (subset[target_column] == target_class).mean()
                    if accuracy > best_accuracy or (accuracy == best_accuracy and coverage > best_coverage):
                        best_condition = condition
                        best_accuracy = accuracy
                        best_coverage = coverage

            if best_condition:
                best_rule['conditions'].append(best_condition)
                best_rule['accuracy'] = best_accuracy
                best_rule['coverage'] = best_coverage
                current_data = current_data.query(best_condition)
            else:
                break
        
        if best_rule['conditions']:
            rules.append(best_rule)
            covered_indices = data.query(' & '.join(best_rule['conditions'])).index
            data = data.drop(covered_indices)

    return rules

# Generate rules for both 'YES' and 'NO' classes
rules_yes = prism(training_data.copy(), target_class='YES')
rules_no = prism(training_data.copy(), target_class='NO')

# Combine all rules
all_rules = rules_yes + rules_no

# Function to apply rules to a dataset
def apply_rules(data, rules):
    def evaluate_row(row):
        for rule in rules:
            # Check if all conditions in the rule hold true for the given row
            if all(eval(f"row['{condition.split()[0]}'] {condition[len(condition.split()[0]):]}") for condition in rule['conditions']):
                return 'YES' if rule in rules_yes else 'NO'
        return 'NO'

    return data.apply(evaluate_row, axis=1)

# Apply rules to validation data and evaluate performance
validation_data['Predicted_Next_Day_Price_Movement'] = apply_rules(validation_data.copy(), all_rules)

# Calculate accuracy
accuracy = (validation_data['Predicted_Next_Day_Price_Movement'] == validation_data['Next_Day_Price_Movement']).mean()
print(f"Validation Accuracy: {accuracy:.2f}")

# Save rules to a text file
rules_output = []
for rule in all_rules:
    rules_output.append(f"If {' and '.join(rule['conditions'])}, then Predict {('YES' if rule in rules_yes else 'NO')}")

with open("rules.txt", "w") as file:
    for rule in rules_output:
        file.write(rule + "\n")

print("Rules saved to 'rules.txt'.")

# Save the Python program that reads the rules and accepts a csv file
rule_based_program = """
import pandas as pd

# Load rules from file
def load_rules(file_path):
    rules = []
    with open(file_path, 'r') as file:
        for line in file:
            conditions = line.strip().split('If ')[-1].split(' then Predict ')[0].split(' and ')
            prediction = line.strip().split(' then Predict ')[-1]
            rules.append({'conditions': conditions, 'prediction': prediction})
    return rules

# Apply rules to a dataset
def apply_rules(data, rules):
    def evaluate_row(row):
        for rule in rules:
            if all(eval(f"row['{condition.split()[0]}'] {condition[len(condition.split()[0]):]}") for condition in rule['conditions']):
                return rule['prediction']
        return 'NO'

    return data.apply(evaluate_row, axis=1)

# Main function to read csv and apply rules
def main(input_csv, rules_file):
    data = pd.read_csv(input_csv)
    rules = load_rules(rules_file)
    data['Predicted_Next_Day_Price_Movement'] = apply_rules(data, rules)
    data.to_csv("output_predictions.csv", index=False)
    print("Predictions saved to 'output_predictions.csv'.")

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1]
    rules_file = sys.argv[2]
    main(input_csv, rules_file)
"""

# Save the rule-based program to a file
with open("rule_based_program.py", "w") as file:
    file.write(rule_based_program)

print("Rule-based Python program saved to 'rule_based_program.py'.")
