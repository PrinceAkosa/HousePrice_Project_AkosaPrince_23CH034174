# Titanic Survival Prediction - Model Development
# Selected Features: Pclass, Sex, Age, Fare, Embarked

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Titanic dataset
print("Loading Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")

# 2. Data Preprocessing

# a. Select only required features (5 input features + 1 target)
# Selected features: Pclass, Sex, Age, Fare, Embarked
selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']
df = df[selected_features]

print(f"\nSelected features: {selected_features}")
print(f"\nMissing values before handling:\n{df.isnull().sum()}")

# b. Handling missing values
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Fare with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print(f"\nMissing values after handling:\n{df.isnull().sum()}")

# c. Encoding categorical variables
# Encode Sex (male=1, female=0)
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Encode Embarked (C=0, Q=1, S=2)
le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

print(f"\nData after encoding:\n{df.head()}")

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# d. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for better readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 3. Implement Random Forest Classifier
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2
)

# 4. Train the model
model.fit(X_train, y_train)
print("Model training completed!")

# 5. Evaluate the model
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:\n{feature_importance}")

# 6. Save the trained model and preprocessing objects
print("\n" + "="*50)
print("Saving model and preprocessing objects...")
print("="*50)

# Create a dictionary with model and all preprocessing objects
model_package = {
    'model': model,
    'scaler': scaler,
    'label_encoders': {
        'Sex': le_sex,
        'Embarked': le_embarked
    },
    'feature_names': X.columns.tolist()
}

joblib.dump(model_package, 'titanic_survival_model.pkl')
print("✓ Model saved as 'titanic_survival_model.pkl'")

# 7. Demonstrate model reloading and prediction
print("\n" + "="*50)
print("DEMONSTRATING MODEL RELOAD AND PREDICTION")
print("="*50)

# Load the saved model
loaded_package = joblib.load('titanic_survival_model.pkl')
loaded_model = loaded_package['model']
loaded_scaler = loaded_package['scaler']

print("✓ Model successfully reloaded!")

# Test prediction with sample data
sample_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # male
    'Age': [22],
    'Fare': [7.25],
    'Embarked': [2]  # S
})

sample_scaled = loaded_scaler.transform(sample_passenger)
prediction = loaded_model.predict(sample_scaled)
probability = loaded_model.predict_proba(sample_scaled)

print(f"\nSample Passenger Details:")
print(f"  - Class: 3rd")
print(f"  - Sex: Male")
print(f"  - Age: 22")
print(f"  - Fare: $7.25")
print(f"  - Embarked: Southampton")
print(f"\nPrediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Probability: {probability[0][prediction[0]]:.2%}")

print("\n" + "="*50)
print("Model development completed successfully!")
print("="*50)