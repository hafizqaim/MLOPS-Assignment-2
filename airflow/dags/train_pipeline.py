from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Default arguments for the DAG
default_args = {
    'owner': 'f223142',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'train_pipeline',
    default_args=default_args,
    description='MLOps training pipeline with Airflow',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'training', 'machine-learning'],
)


def load_data_task(**context):
    """Task 1: Load dataset"""
    print("Loading dataset from data/dataset.csv")
    
    # Use the mounted volume path
    data_path = '/opt/airflow/data/dataset.csv'
    
    print(f"Attempting to read from: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")
    
    data = pd.read_csv(data_path)
    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Push data info to XCom
    context['ti'].xcom_push(key='dataset_shape', value=str(data.shape))
    context['ti'].xcom_push(key='dataset_path', value=data_path)
    
    return "Data loaded successfully"


def preprocess_and_train_model_task(**context):
    """Task 2: Preprocess data and train model"""
    print("Starting model training...")
    
    # Get dataset path from previous task
    data_path = context['ti'].xcom_pull(task_ids='load_data', key='dataset_path')
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Preprocess
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained! Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model to disk immediately (avoid XCom serialization issues)
    models_dir = '/opt/airflow/models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model_airflow.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Push metrics to XCom (JSON serializable)
    context['ti'].xcom_push(key='accuracy', value=accuracy)
    context['ti'].xcom_push(key='train_size', value=len(X_train))
    context['ti'].xcom_push(key='test_size', value=len(X_test))
    context['ti'].xcom_push(key='model_path', value=model_path)
    
    return "Model trained successfully"


def save_model_task(**context):
    """Task 3: Verify model was saved"""
    print("Verifying model was saved...")
    
    # Get model path from previous task
    model_path = context['ti'].xcom_pull(task_ids='train_model', key='model_path')
    
    # Verify file exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✓ Model file exists at: {model_path}")
        print(f"✓ Model file size: {file_size} bytes")
        
        # Test loading the model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✓ Model successfully loaded and verified")
        print(f"✓ Model type: {type(loaded_model).__name__}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return "Model verified successfully"


def log_results_task(**context):
    """Task 4: Log final results"""
    print("=" * 50)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 50)
    
    # Pull metrics from previous tasks
    dataset_shape = context['ti'].xcom_pull(task_ids='load_data', key='dataset_shape')
    accuracy = context['ti'].xcom_pull(task_ids='train_model', key='accuracy')
    train_size = context['ti'].xcom_pull(task_ids='train_model', key='train_size')
    test_size = context['ti'].xcom_pull(task_ids='train_model', key='test_size')
    model_path = context['ti'].xcom_pull(task_ids='save_model', key='model_path')
    
    print(f"Dataset Shape: {dataset_shape}")
    print(f"Train Size: {train_size}")
    print(f"Test Size: {test_size}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model saved at: {model_path}")
    print("=" * 50)
    
    return "Pipeline completed successfully"


# Define tasks
load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=preprocess_and_train_model_task,
    dag=dag,
)

save_model = PythonOperator(
    task_id='save_model',
    python_callable=save_model_task,
    dag=dag,
)

log_results = PythonOperator(
    task_id='log_results',
    python_callable=log_results_task,
    dag=dag,
)

# Set task dependencies
load_data >> train_model >> save_model >> log_results
