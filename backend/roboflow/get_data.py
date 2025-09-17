import os
import logging
from roboflow import Roboflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")  # load from env
WORKSPACE = "test-mm1q9"
PROJECT = "skinsafeai-3mupa"
VERSION = 2
FORMAT = "yolov8"

def download_dataset():
    """Download a dataset from Roboflow."""
    try:
        if not ROBOFLOW_API_KEY:
            raise ValueError("Missing ROBOFLOW_API_KEY in environment variables")

        logging.info("Connecting to Roboflow...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)

        logging.info(f"Accessing project {PROJECT} (v{VERSION}) in workspace {WORKSPACE}...")
        project = rf.workspace(WORKSPACE).project(PROJECT)
        version = project.version(VERSION)

        logging.info(f"Downloading dataset in {FORMAT} format...")
        dataset = version.download(FORMAT)
        logging.info("Download complete.")
        return dataset

    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_dataset()
