# Text-Summarization: A Production-Grade NLP Summarization System

An end-to-end, NLP-driven pipeline for automated text summarization engineered for modern content processing needs. By leveraging state-of-the-art Hugging Face Transformers, PyTorch acceleration, and modular MLOps practices, this system delivers accurate, concise summaries—ready for production deployment at scale.

---

## 📂 Repository Structure
```
.
├── .github/workflows/ # CI/CD automation pipelines
├── config/ # Project-wide YAML configuration
├── notebook/ # Exploratory analysis & experiments
├── src/
│ └── text_summarization/ # Package source
│ ├── configuration.py # Config dataclasses
│ ├── components/ # Data ingestion, preprocessing, training modules
│ ├── pipeline/ # Ingestion, transformation, training, prediction
│ ├── utils/ # Model helpers, file I/O utilities
│ ├── logger.py # Structured logging setup
│ └── exception.py # Custom exception handling
├── Dockerfile # Container specification
├── dvc.yaml / dvc.lock # DVC pipeline stages & lock file
├── params.json # Training hyperparameters
├── requirements.txt # Python dependencies
└── setup.py # Package installer
```

---

## 🔧 Core Workflow

1. **Data Ingestion**  
   Uses DVC to pull training datasets (news articles, documents) from remote storage, running `stage_01_data_ingestion.py` to persist cleaned and tokenized datasets.

2. **Data Preprocessing & Tokenization**  
   Tokenizes text using SentencePiece and Hugging Face tokenizers in `stage_02_data_preprocessing.py`, preparing input sequences for transformer models.

3. **Model Training & Fine-Tuning**  
   Fine-tunes pre-trained transformer models (T5, BART, or custom architectures) through `stage_03_model_trainer.py` using PyTorch with distributed training acceleration via `accelerate`.

4. **Evaluation & Metrics**  
   Evaluates model performance using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) and Hugging Face `evaluate` library in `stage_04_model_evaluation.py`, generating comprehensive performance reports.

5. **Real-Time Inference API**  
   - **FastAPI Backend** (`app.py`): Exposes `/summarize` and `/train` REST endpoints with Uvicorn ASGI server for high-performance inference.
   - **Response Times**: Sub-second summarization for documents up to 10,000 words.
   - **Deployment**: Production-ready containerized service with health checks and logging.

---

## ✅ Key Capabilities

- **State-of-the-Art NLP Models**  
  Leverages Hugging Face Transformers (v4.49.0) with support for T5, BART, PEGASUS, and custom fine-tuned models for domain-specific summarization.

- **Modular, Scalable Architecture**  
  Decoupled data ingestion, preprocessing, training, and inference layers enable easy model swapping and horizontal scaling.

- **Production-Ready MLOps Stack**  
  - **DVC** for dataset and model versioning with S3 backend integration
  - **Configuration-driven development** using YAML for reproducible experiments
  - **Structured logging** for pipeline transparency and debugging
  - **Custom exception handling** for robust error management

- **Comprehensive Evaluation**  
  Built-in ROUGE score calculation (precision, recall, F1) with support for custom evaluation metrics and automated performance tracking.

- **Distributed Training Support**  
  Utilizes Hugging Face `accelerate` library for efficient multi-GPU training and mixed-precision optimization.

- **Extensible Architecture**  
  Easily swap transformer architectures, tokenizers, or add custom preprocessing steps with minimal code changes.

---

## 🚀 Deployment & CI/CD

- **GitHub Actions**  
  Automated linting, testing, Docker builds, and deployment pipelines triggered on every commit (`.github/workflows/`).

- **AWS EC2 Deployment**  
  - Deployed on AWS EC2 instances with auto-scaling capabilities
  - Docker containerization for consistent environments
  - AWS ECR for secure image storage and version management
  - Environment variable management via `.env` files

- **Self-Hosted GitHub Runners**  
  Configured Linux-based self-hosted runners for cost-effective CI/CD automation with background service management.

- **Environment-Driven Configuration**  
  Manage secrets and endpoints via `.env` file:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
AWS_ECR_LOGIN_URI=your_ecr_uri
ECR_REPOSITORY_NAME=your_repo_name
```

---

## 🏃 Running Locally

1. **Clone the repository**
```
git clone https://github.com/hasan-raza-01/Text-Summarization.git
cd Text-Summarization
```
2. **Set up environment variables**
```
cp .env.example .env
```
#### Edit .env with your AWS credentials and configuration
3. **Create virtual environment & install dependencies**
```
pip install --upgrade pip uv
uv venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/Mac
uv pip install -e .
```
4. **Run the complete pipeline**
#### Execute full DVC pipeline (data ingestion → training → evaluation)
```
dvc repro
```
5. **Start FastAPI application**
```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
6. **(Alternative) Docker deployment**
```
docker build -t text-summarization:latest .
docker run -p 8000:8000 --env-file .env text-summarization:latest
```
7. **Test the API**  
Navigate to `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## 🛠️ Technology Stack

- **NLP & Transformers**: Hugging Face Transformers (v4.49.0), PyTorch (v2.6.0)
- **Evaluation**: ROUGE metrics (rouge-score, rouge), Hugging Face Evaluate
- **MLOps**: DVC (v3.59.0), DVC-S3 integration
- **Backend**: FastAPI (v0.115.8), Uvicorn (v0.34.0)
- **Training Optimization**: Accelerate (v1.4.0), SentencePiece tokenization
- **Deployment**: Docker, AWS EC2, AWS ECR, GitHub Actions
- **Development**: Python-dotenv, Python-box, PyYAML, tqdm

---

## 📝 Workflow Configuration

The project follows a structured workflow pattern:

1. Update `config.yaml` with dataset paths and model parameters
2. Update `.env` (optional) with AWS credentials
3. Update constants in `src/text_summarization/constants.py`
4. Update entity definitions for data structures
5. Update configuration classes
6. Update component implementations (data ingestion, training, etc.)
7. Update pipeline orchestration
8. Update `main.py` for end-to-end execution
9. Update `dvc.yaml` for pipeline stage definitions

---

## Setup GitHub Secrets

1. AWS_ACCESS_KEY_ID
2. AWS_SECRET_ACCESS_KEY
3. AWS_REGION
4. AWS_ECR_LOGIN_URI
5. ECR_REPOSITORY_NAME

---

## Docker Setup In EC2 Commands to be Executed

### Optional
```
sudo apt-get update -y
sudo apt-get upgrade -y
```
### Required
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

---

## Add Environment Variables to EC2 Instance

### Create/edit the .env file:
```
nano /home/ubuntu/.env
```
#### Paste all variables and press CTRL+X, Y, ENTER

---

## Reconnect with GitHub Runner
```
cd actions-runner && ./run.sh
```
