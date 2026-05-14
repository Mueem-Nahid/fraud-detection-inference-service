# ModelServe

ModelServe is a capstone MLOps platform for serving a fraud-detection model through FastAPI. It combines MLflow model registry, Feast online feature retrieval with Redis, PostgreSQL-backed experiment tracking, Prometheus metrics, Grafana dashboards, Pulumi-provisioned AWS infrastructure, and GitHub Actions CI/CD.

The project prioritizes a reproducible serving stack and clear operations over model accuracy. The baseline model uses the Kaggle Credit Card Transactions Fraud Detection dataset and is intentionally simple enough to retrain during a lab/demo session.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- AWS CLI configured for the sandbox account
- Pulumi CLI
- GitHub repository secrets for CI/CD deployment

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Place fraudTrain.csv in training/, then train and register the model.
docker compose up -d postgres redis mlflow
python training/train.py

# Materialize features after Redis is running.
python scripts/materialize_features.py

# Start the remaining services.
docker compose up -d

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @training/sample_request.json
```

Local service URLs:

| Service | URL |
| --- | --- |
| FastAPI | `http://localhost:8000` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

Default Grafana login is `admin` / `admin123` unless overridden.

## REST Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Returns service health and the loaded model version. |
| POST | `/predict` | Accepts `{"entity_id": <int>}` and returns prediction, probability, model version, and timestamp. |
| GET | `/predict/{entity_id}?explain=true` | Returns a prediction plus the feature values used for debugging. |
| GET | `/metrics` | Exposes Prometheus metrics including request count, latency, errors, model version, and Feast hit/miss counters. |

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `MODEL_NAME` | MLflow registered model name loaded by the API. | `fraud_detection_model` |
| `MODEL_STAGE` | MLflow model stage to serve. | `Production` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL. | `http://mlflow:5000` |
| `FEAST_REPO_PATH` | Path to the Feast repository inside the API container. | `/app/feast_repo` |
| `REDIS_HOST` | Redis host for Feast online store. | `redis` |
| `REDIS_PORT` | Redis port. | `6379` |
| `POSTGRES_HOST` | PostgreSQL host for MLflow backend store. | `postgres` |
| `POSTGRES_PORT` | PostgreSQL port. | `5432` |
| `POSTGRES_USER` | PostgreSQL username. | `mlflow` |
| `POSTGRES_PASSWORD` | PostgreSQL password. | `mlflow_password` |
| `POSTGRES_DB` | PostgreSQL database name. | `mlflow` |
| `MLFLOW_BACKEND_STORE_URI` | SQLAlchemy URI for MLflow backend storage. | `postgresql://mlflow:mlflow_password@postgres:5432/mlflow` |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | MLflow artifact root, local path or S3 URI. | `/mlflow/artifacts` |
| `AWS_REGION` | AWS region for Pulumi, ECR, and S3. | `ap-southeast-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key for local/CI AWS calls. | unset |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for local/CI AWS calls. | unset |
| `MLFLOW_S3_ENDPOINT_URL` | Optional custom S3 endpoint. | unset |
| `GF_SECURITY_ADMIN_USER` | Grafana admin username. | `admin` |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password. | `admin123` |
| `GF_USERS_ALLOW_SIGN_UP` | Enables/disables Grafana signup. | `false` |
| `API_IMAGE` | Optional image override used by EC2 deployment. | `modelserve-api:local` |

## GitHub Secrets

| Secret | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | Allows CI to authenticate to AWS. |
| `AWS_SECRET_ACCESS_KEY` | Allows CI to authenticate to AWS. |
| `AWS_ACCOUNT_ID` | Builds the ECR registry URL. |
| `EC2_SSH_KEY` | Private SSH key for the deployment EC2 instance. |
| `EC2_HOST` | Public IP or DNS name of the deployment EC2 instance. |
| `EC2_USERNAME` | SSH user for the EC2 instance, usually `ec2-user`. |
| `GRAFANA_ADMIN_PASSWORD` | Optional Grafana admin password used by CI deployment. |

Repository variables:

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` | Overrides the default `ap-southeast-1` region. |
| `ECR_IMAGE_NAME` | Overrides the default ECR image name `modelserve`. |

## Infrastructure

Pulumi provisions a VPC, public subnet, internet gateway, route table, security group, EC2 instance, ECR repository, S3 bucket, and IAM role/profile tagged with `Project: modelserve`.

```bash
cd infrastructure
pip install -r requirements.txt
pulumi stack init dev
pulumi up --yes
pulumi stack output
```

Destroy resources after the lab/demo:

```bash
pulumi destroy --yes
```

## Documentation

Read the full engineering documentation, ADRs, runbook, and limitations in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Dataset: [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection). Place `fraudTrain.csv` under `training/` before running `training/train.py`.
