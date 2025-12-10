import pytest
import pandas as pd
import pickle
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train import load_data, preprocess_data, train_model, save_model

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_data_file_exists(self):
        """Test that dataset file exists"""
        assert os.path.exists('data/dataset.csv'), "Dataset file not found"
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame"""
        data = load_data('data/dataset.csv')
        assert isinstance(data, pd.DataFrame), "load_data should return a DataFrame"
    
    def test_data_shape(self):
        """Test that dataset has expected number of columns"""
        data = load_data('data/dataset.csv')
        assert data.shape[1] == 10, f"Expected 10 columns, got {data.shape[1]}"
    
    def test_data_not_empty(self):
        """Test that dataset is not empty"""
        data = load_data('data/dataset.csv')
        assert len(data) > 0, "Dataset should not be empty"
    
    def test_required_columns_exist(self):
        """Test that all required columns are present"""
        data = load_data('data/dataset.csv')
        required_columns = [
            'customer_id', 'age', 'income', 'credit_score', 
            'account_balance', 'tenure_months', 'num_products',
            'has_credit_card', 'is_active_member', 'churn'
        ]
        for col in required_columns:
            assert col in data.columns, f"Column '{col}' not found in dataset"


class TestDataPreprocessing:
    """Test data preprocessing functionality"""
    
    def test_preprocess_data_separates_features_target(self):
        """Test that preprocessing correctly separates features and target"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        assert isinstance(X, pd.DataFrame), "Features should be a DataFrame"
        assert isinstance(y, pd.Series), "Target should be a Series"
    
    def test_target_column_removed_from_features(self):
        """Test that target column is not in features"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        assert 'churn' not in X.columns, "Target column should not be in features"
        assert 'customer_id' not in X.columns, "ID column should not be in features"
    
    def test_feature_count(self):
        """Test that we have the correct number of features"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        # 10 original columns - 2 (customer_id, churn) = 8 features
        assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"


class TestModelTraining:
    """Test model training functionality"""
    
    def test_train_model_returns_model(self):
        """Test that train_model returns a trained model"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        # Use small subset for quick testing
        X_train = X[:20]
        y_train = y[:20]
        
        model = train_model(X_train, y_train)
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
    
    def test_model_can_predict(self):
        """Test that trained model can make predictions"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        X_train = X[:20]
        y_train = y[:20]
        X_test = X[20:]
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Prediction length should match test data"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be 0 or 1"
    
    def test_model_prediction_shape(self):
        """Test that model predictions have correct shape"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        X_train = X[:20]
        y_train = y[:20]
        X_test = X[20:25]
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape[0] == X_test.shape[0], "Prediction count mismatch"


class TestModelSaving:
    """Test model saving functionality"""
    
    def test_save_model_creates_file(self):
        """Test that save_model creates a pickle file"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        X_train = X[:20]
        y_train = y[:20]
        
        model = train_model(X_train, y_train)
        
        # Save to test location
        test_path = 'models/test_model.pkl'
        save_model(model, test_path)
        
        assert os.path.exists(test_path), "Model file should be created"
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
    
    def test_saved_model_can_be_loaded(self):
        """Test that saved model can be loaded and used"""
        data = load_data('data/dataset.csv')
        X, y = preprocess_data(data)
        
        X_train = X[:20]
        y_train = y[:20]
        X_test = X[20:25]
        
        model = train_model(X_train, y_train)
        
        # Save and load model
        test_path = 'models/test_model.pkl'
        save_model(model, test_path)
        
        with open(test_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test loaded model can predict
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(X_test), "Loaded model should make predictions"
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
