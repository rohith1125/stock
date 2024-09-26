import pandas as pd
import numpy as np

# 1. Load and Prepare the Data
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['Next_Open'] = data['Open'].shift(-1)  # Create next day's opening price
    data['Target'] = (data['Next_Open'] > data['Close']).astype(int)  # 1 if next day's open > today's close
    data.loc[199, 'Target'] = 1  # Given: 200th day's true value is YES (1)
    return data.iloc[:-1]  # Drop last row as it doesn't have a target

# 2. Generate Initial Rules
def generate_initial_rules(data, target_column, min_coverage=5, min_accuracy=0.6):
    rules = []
    features = [col for col in data.columns if col not in ['Target', 'Next_Open', 'Unnamed: 0']]
    
    for feature in features:
        # Use quantile values to create different thresholds
        thresholds = data[feature].quantile([0.25, 0.5, 0.75]).values
        for threshold in thresholds:
            for operator in ['>', '<']:
                rule = [(feature, operator, threshold)]
                coverage, accuracy = calculate_coverage_accuracy(data, rule, target_column)
                if coverage >= min_coverage and accuracy >= min_accuracy:
                    rules.append((rule, coverage, accuracy))
    return rules

# 3. Calculate Coverage and Accuracy
def calculate_coverage_accuracy(data, rule, target_column):
    condition = np.ones(len(data), dtype=bool)
    for feature, operator, threshold in rule:
        if operator == '>':
            condition &= data[feature] > threshold
        elif operator == '<':
            condition &= data[feature] < threshold

    covered_data = data[condition]
    coverage = len(covered_data)
    if coverage == 0:
        return 0, 0  # No coverage, no accuracy
    accuracy = sum(covered_data[target_column]) / coverage
    return coverage, accuracy

# 4. Refine and Combine Rules
def refine_and_combine_rules(data, initial_rules, target_column, min_coverage=5, min_accuracy=0.6):
    refined_rules = []
    features = [col for col in data.columns if col not in ['Target', 'Next_Open', 'Unnamed: 0']]
    
    for rule, base_coverage, base_accuracy in initial_rules:
        if base_coverage < min_coverage or base_accuracy < min_accuracy:
            continue  # Skip if it doesn't meet basic criteria
        
        for feature in features:
            if any(cond[0] == feature for cond in rule):
                continue  # Skip if feature already in rule
            
            thresholds = data[feature].quantile([0.25, 0.5, 0.75]).values
            for threshold in thresholds:
                for operator in ['>', '<']:
                    new_rule = rule + [(feature, operator, threshold)]
                    coverage, accuracy = calculate_coverage_accuracy(data, new_rule, target_column)
                    if coverage >= min_coverage and accuracy >= min_accuracy:
                        refined_rules.append((new_rule, coverage, accuracy))
    
    return refined_rules

# 5. Apply Rules to Predict
def apply_rules(data, rules):
    predictions = []
    for _, row in data.iterrows():
        prediction = 0  # Default prediction is 'NO'
        for rule, _, _ in rules:
            condition = True
            for feature, operator, threshold in rule:
                if operator == '>':
                    condition &= row[feature] > threshold
                elif operator == '<':
                    condition &= row[feature] < threshold
            if condition:
                prediction = 1  # Rule matches, predict 'YES'
                break
        predictions.append(prediction)
    return predictions

# 6. Evaluate Rules on Validation Data
def evaluate_rules_on_validation(validation_file, rules):
    validation_data = pd.read_csv(validation_file)
    validation_data['Prediction'] = apply_rules(validation_data, rules)
    return validation_data

# Main Function
def main():
    # Load and prepare the training data
    training_file = "Training200.csv"
    validation_file = "Validation65.csv"
    training_data = load_and_prepare_data(training_file)
    
    # Generate initial rules
    initial_rules = generate_initial_rules(training_data, 'Target')
    print(f"Generated {len(initial_rules)} initial rules.")
    
    # Refine and combine rules
    refined_rules = refine_and_combine_rules(training_data, initial_rules, 'Target')
    refined_rules.sort(key=lambda x: (-x[2], -x[1]))  # Sort by accuracy then coverage
    print(f"Generated {len(refined_rules)} refined rules.")
    
    # Display top rules
    for idx, (rule, coverage, accuracy) in enumerate(refined_rules[:5], 1):
        print(f"Rule {idx} (Coverage: {coverage}, Accuracy: {accuracy}):")
        for feature, operator, threshold in rule:
            print(f"  {feature} {operator} {threshold}")
            
    # Evaluate rules on validation data
    validation_results = evaluate_rules_on_validation(validation_file, refined_rules)
    print(validation_results.head())
    
    # Save validation predictions
    validation_results.to_csv("Validation_Predictions.csv", index=False)
    print("Validation predictions saved to 'Validation_Predictions.csv'.")

# Run the main function
if __name__ == "__main__":
    main()
