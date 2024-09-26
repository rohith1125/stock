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