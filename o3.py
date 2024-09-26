# Import necessary libraries
import pandas as pd
import numpy as np

# Load the training and validation data
training_data = pd.read_csv('Training200.csv')
validation_data = pd.read_csv('Validation65.csv')

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in training_data.columns:
    training_data = training_data.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in validation_data.columns:
    validation_data = validation_data.drop(columns=['Unnamed: 0'])

# Create the target variable for the training data
NextDayUp_train = []
for i in range(len(training_data) - 1):
    if training_data.loc[i + 1, 'Open'] > training_data.loc[i, 'Close']:
        NextDayUp_train.append(1)
    else:
        NextDayUp_train.append(0)
NextDayUp_train.append(1)  # Assuming last day is 'Yes' as per your note
training_data['NextDayUp'] = NextDayUp_train

# Create the target variable for the validation data
NextDayUp_valid = []
for i in range(len(validation_data) - 1):
    if validation_data.loc[i + 1, 'Open'] > validation_data.loc[i, 'Close']:
        NextDayUp_valid.append(1)
    else:
        NextDayUp_valid.append(0)
NextDayUp_valid.append(1)  # Assuming last day is 'Yes'
validation_data['NextDayUp'] = NextDayUp_valid

# Remove last row if target cannot be computed
training_data.dropna(inplace=True)
validation_data.dropna(inplace=True)

# Handle categorical features
for col in ['Dividends', 'Stock Splits']:
    if col in training_data.columns:
        training_data[col] = training_data[col].astype(str)
        validation_data[col] = validation_data[col].astype(str)

# One-hot encode categorical features
categorical_features = training_data.select_dtypes(include=['object']).columns
training_data = pd.get_dummies(training_data, columns=categorical_features)
validation_data = pd.get_dummies(validation_data, columns=categorical_features)

# Align columns
training_data, validation_data = training_data.align(validation_data, join='left', axis=1, fill_value=0)

# Prepare features and target
X_train = training_data.drop(columns=['NextDayUp'])
y_train = training_data['NextDayUp']
X_valid = validation_data.drop(columns=['NextDayUp'])
y_valid = validation_data['NextDayUp']

# Import the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize and train the model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_valid)
y_proba = clf.predict_proba(X_valid)[:, 1]

# Evaluate performance
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred, zero_division=0)
recall = recall_score(y_valid, y_pred, zero_division=0)
f1 = f1_score(y_valid, y_pred, zero_division=0)
auc_score = roc_auc_score(y_valid, y_proba)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"ROC AUC Score: {auc_score:.2f}")

# Feature importance
importances = clf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df.head(10))
