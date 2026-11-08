import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
from scripts.task3.mlflow_toy_local import run_toy_experiment

def main():
    print("=" * 60)
    print(" TASK 3.2: REPEAT TOY RUN AGAINST DAGSHUB MLFLOW")
    print("=" * 60)
    
    # Load credentials strictly from .env
    load_dotenv(".env")
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if not tracking_uri or not username or not password:
        raise ValueError("Missing MLFLOW credentials in .env file!")
        
    print(f"Tracking URI: {tracking_uri}")
    print(f"Username    : {username}")
    print(f"Token/Pass  : {'*' * len(password)}")
    print("-" * 60)
    
    # Ensure environment variables are set for MLflow HTTP client
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    
    # Run against DagsHub
    run_toy_experiment(tracking_uri=tracking_uri, experiment_name="Toy_MLflow_Lab_DagsHub")

if __name__ == "__main__":
    main()
