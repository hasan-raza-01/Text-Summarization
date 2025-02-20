from text_summarization.constants import (
    DataIngestionConstants,
    DataTransformationConstants,
    ModelTrainerConstants
)
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class ModelTrainerConfig:
    ARITFACTS_ROOT_DIR_PATH = os.path.join(ModelTrainerConstants.ARITFACTS_ROOT_DIR_NAME)
    MODEL_ROOT_DIR_PATH = os.path.join(ARITFACTS_ROOT_DIR_PATH, ModelTrainerConstants.MODEL_ROOT_DIR_NAME)
    TRAINER_ROOT_DIR_PATH = os.path.join(MODEL_ROOT_DIR_PATH, ModelTrainerConstants.TRAINER_ROOT_DIR_NAME)
    BASE_ESTIMATOR_PATH = os.path.join(TRAINER_ROOT_DIR_PATH, ModelTrainerConstants.BASE_ESTIMATOR_NAME)
    FINETUNED_ESTIMATOR_PATH = os.path.join(TRAINER_ROOT_DIR_PATH, ModelTrainerConstants.FINETUNED_ESTIMATOR_NAME)
    PARAMS_FILE_PATH = Path(ModelTrainerConstants.PARAMS_FILE_NAME)


