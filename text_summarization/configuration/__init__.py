from text_summarization.constants import (
    DataIngestionConstants,
    DataTransformationConstants
)
from dataclasses import dataclass
import os


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DataTransformationConfig:
    ARITFACTS_ROOT_DIR_PATH=os.path.join(DataTransformationConstants.ARITFACTS_ROOT_DIR_NAME)
    DATA_ROOT_DIR_PATH=os.path.join(ARITFACTS_ROOT_DIR_PATH, DataTransformationConstants.DATA_ROOT_DIR_NAME)
    TRANSFORMATION_ROOT_DIR_PATH=os.path.join(DATA_ROOT_DIR_PATH, DataTransformationConstants.TRANSFORMATION_ROOT_DIR_NAME)
    TRAIN_DATA_DIR_PATH=os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TRAIN_DATA_DIR_NAME)
    VALIDATION_DATA_DIR_PATH=os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.VALIDATION_DATA_DIR_NAME)
    TEST_DATA_DIR_PATH=os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TEST_DATA_DIR_NAME)
    TOKENIZER_PATH=os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TOKENIZER_NAME)
    MODEL_REPO_ID=DataTransformationConstants.MODEL_REPO_ID


