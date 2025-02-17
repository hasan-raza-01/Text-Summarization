from text_summarization.constants import (
    DataIngestionConstants
)
from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    # DIR'S
    ARITFACTS_ROOT_DIR_PATH = os.path.join(DataIngestionConstants.ARITFACTS_ROOT_DIR_NAME)
    DATA_ROOT_DIR_PATH = os.path.join(ARITFACTS_ROOT_DIR_PATH, DataIngestionConstants.DATA_ROOT_DIR_NAME)
    INGESTION_ROOT_DIR_PATH = os.path.join(DATA_ROOT_DIR_PATH, DataIngestionConstants.INGESTION_ROOT_DIR_NAME)
    FEATURE_STORE_ROOT_DIR_PATH = os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.FEATURE_STORE_ROOT_DIR_NAME)
    INGESTED_ROOT_DIR_PATH = os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.INGESTED_ROOT_DIR_NAME)
    # FILES
    ZIP_FILE_PATH = os.path.join(FEATURE_STORE_ROOT_DIR_PATH, DataIngestionConstants.ZIP_FILE_NAME)
    # URI'S
    SOURCE_URI = DataIngestionConstants.SOURCE_URI


