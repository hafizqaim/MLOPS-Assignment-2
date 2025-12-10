import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data(filepath):
    """Load dataset from CSV file"""
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    return data

def preprocess_data(data):
    """Prepare features and target variable"""
    # Separate features and target
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def train_model(X_train, y_train):
    """Train Random Forest Classifier"""
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, filepath):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")

def main():
    # Load dataset
    data = load_data('data/dataset.csv')
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, 'models/model.pkl')
    
    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
