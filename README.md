# MLOps Assignment 02 - Customer Churn Prediction

**Student ID:** f223142  
**Repository:** [MLOPS-Assignment-2](https://github.com/hafizqaim/MLOPS-Assignment-2)  
**Docker Hub:** [hafizqaim](https://hub.docker.com/u/hafizqaim)

## 📋 Project Overview

This project demonstrates a complete MLOps pipeline for customer churn prediction, including:
- Version control with Git & DVC
- CI/CD automation with GitHub Actions
- Containerization with Docker
- Workflow orchestration with Apache Airflow
- REST API deployment with FastAPI
- Cloud deployment on AWS EC2

## 🏗️ Project Structure

```
22F-3142_Ass#2/
├── .github/workflows/
│   └── ci.yml                 # CI/CD pipeline with automated tests
├── airflow/
│   ├── dags/
│   │   └── train_pipeline.py  # Airflow DAG for ML training
│   ├── logs/                  # Airflow execution logs
│   └── data/                  # Data directory for Airflow
├── api/
│   └── main.py                # FastAPI application
├── data/
│   ├── dataset.csv            # Training dataset (tracked by DVC)
│   └── dataset.csv.dvc        # DVC tracking file
├── models/
│   ├── model.pkl              # Trained model from DVC pipeline
│   └── model_airflow.pkl      # Trained model from Airflow
├── src/
│   └── train.py               # Model training script
├── tests/
│   └── test_train.py          # Unit tests for training pipeline
├── docker-compose.yml         # Airflow orchestration setup
├── Dockerfile                 # Training container image
├── Dockerfile.api             # API container image
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # DVC pipeline lock file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git
- DVC 3.x

### 1. Clone Repository
```bash
git clone https://github.com/hafizqaim/MLOPS-Assignment-2.git
cd MLOPS-Assignment-2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Data with DVC
```bash
dvc pull
```

### 4. Run Training Pipeline
```bash
python src/train.py
```

## 🐳 Docker Usage

### Training Container
```bash
# Build training image
docker build -t hafizqaim/mlops-app:v1 .

# Run training
docker run hafizqaim/mlops-app:v1
```

### API Container
```bash
# Build API image
docker build -f Dockerfile.api -t hafizqaim/mlops-api:v1 .

# Run API
docker run -d -p 8000:8000 hafizqaim/mlops-api:v1

# Test API
curl http://localhost:8000/health
```

### Pull from Docker Hub
```bash
docker pull hafizqaim/mlops-app:v1
docker pull hafizqaim/mlops-api:v1
```

## 🌬️ Airflow Orchestration

### Start Airflow
```bash
docker-compose up -d
```

### Access Airflow Web UI
- URL: http://localhost:9090
- Username: `airflow`
- Password: `airflow`

### Trigger DAG
1. Navigate to http://localhost:9090
2. Enable the `train_model_pipeline` DAG
3. Click "Trigger DAG" to start execution

### Pipeline Tasks
1. **load_data_task** - Load dataset from CSV
2. **preprocess_and_train_model_task** - Train RandomForest model
3. **save_model_task** - Save model to disk
4. **log_results_task** - Log accuracy metrics

## 🌐 FastAPI Endpoints

### Health Check
```bash
GET /health
```
Returns API status and model information.

### Predict Single Customer
```bash
POST /predict
Content-Type: application/json

{
  "age": 35,
  "income": 50000,
  "credit_score": 650,
  "account_balance": 10000,
  "tenure_months": 24,
  "num_products": 2,
  "has_credit_card": 1,
  "is_active_member": 1
}
```

### Predict Batch
```bash
POST /predict/batch
Content-Type: application/json

{
  "customers": [
    { "age": 35, "income": 50000, ... },
    { "age": 42, "income": 75000, ... }
  ]
}
```

### Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ☁️ AWS EC2 Deployment

### EC2 Instance Details
- **Instance Type:** t2.micro
- **OS:** Ubuntu 22.04 LTS
- **Public IP:** 16.16.184.224
- **API URL:** http://16.16.184.224:8000

### Deployment Steps
1. Launch EC2 instance
2. Configure security group (port 8000)
3. SSH into instance
4. Install Docker
5. Pull and run API container

For detailed instructions, see `EC2-DEPLOYMENT.md`.

## 🔄 CI/CD Pipeline

GitHub Actions workflow runs on every push:
1. Checkout code
2. Set up Python 3.10
3. Install dependencies
4. Pull DVC data (with fallback to synthetic data)
5. Run linting with flake8
6. Run unit tests with pytest
7. Report test results

View workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## 📊 DVC Pipeline

### View Pipeline DAG
```bash
dvc dag
```

```
+----------------------+ 
| data\dataset.csv.dvc | 
+----------------------+ 
            *
            *
            *
    +-------------+      
    | train_model |      
    +-------------+
```

### Reproduce Pipeline
```bash
dvc repro
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Linting
```bash
flake8 src/ tests/ api/ --max-line-length=120
```

## 📦 Dependencies

- **ML/Data:** pandas, scikit-learn, numpy
- **API:** fastapi, uvicorn, pydantic, python-multipart
- **Testing:** pytest, flake8
- **Orchestration:** Apache Airflow 2.8.0 (via Docker)
- **Version Control:** DVC 3.x

## 🛠️ Tools & Technologies

| Category | Tool | Purpose |
|----------|------|---------|
| Version Control | Git, DVC | Code and data versioning |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Containerization | Docker | Application packaging |
| Orchestration | Apache Airflow | Workflow automation |
| API Framework | FastAPI | REST API development |
| Cloud Platform | AWS EC2 | Production deployment |
| ML Framework | scikit-learn | Model training |

## 📝 Assignment Tasks

- ✅ **Task 1:** Git & DVC setup with pipeline
- ✅ **Task 2:** CI/CD with GitHub Actions
- ✅ **Task 3:** Docker containerization for training
- ✅ **Task 4:** Airflow pipeline orchestration
- ✅ **Task 5:** FastAPI REST API development
- ✅ **Task 6:** AWS EC2 deployment
- ✅ **Task 7:** Documentation and reflection

## 📄 Additional Documentation

- **REFLECTION_REPORT.md** - Problems faced, fixes applied, and learnings
- **EC2-DEPLOYMENT.md** - Detailed AWS deployment instructions
- **deploy-ec2.sh** - Automated EC2 deployment script

## 🤝 Contributing

This is an academic assignment project. For questions or issues, please contact the repository owner.

## 📅 Project Timeline

**Deadline:** December 14, 2025  
**Status:** ✅ Complete

---

**Developed as part of MLOps Course Assignment**  
**Student ID: f223142**
