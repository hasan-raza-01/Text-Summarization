from text_summarization.constants import (
    DataIngestionConstants,
    DataTransformationConstants,
    ModelTrainerConstants,
    ModelEvaluationConstants
)
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import os

@dataclass
class time_stamp:
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# get time stamp
time_stamp_object = time_stamp()
timestamp = time_stamp_object.time

@dataclass(frozen=True)
class DataIngestionConfig:
    __timestamp = timestamp
    # DIR'S
    ARITFACTS_ROOT_DIR_PATH = Path(os.path.join(DataIngestionConstants.ARITFACTS_ROOT_DIR_NAME, __timestamp))
    DATA_ROOT_DIR_PATH = Path(os.path.join(ARITFACTS_ROOT_DIR_PATH, DataIngestionConstants.DATA_ROOT_DIR_NAME))
    INGESTION_ROOT_DIR_PATH = Path(os.path.join(DATA_ROOT_DIR_PATH, DataIngestionConstants.INGESTION_ROOT_DIR_NAME))
    FEATURE_STORE_ROOT_DIR_PATH = Path(os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.FEATURE_STORE_ROOT_DIR_NAME))
    INGESTED_ROOT_DIR_PATH = Path(os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.INGESTED_ROOT_DIR_NAME))
    # FILES
    ZIP_FILE_PATH = Path(os.path.join(FEATURE_STORE_ROOT_DIR_PATH, DataIngestionConstants.ZIP_FILE_NAME))
    # URI'S
    SOURCE_URI = DataIngestionConstants.SOURCE_URI



@dataclass(frozen=True)
class DataTransformationConfig:
    __timestamp = timestamp
    ARITFACTS_ROOT_DIR_PATH=Path(os.path.join(DataTransformationConstants.ARITFACTS_ROOT_DIR_NAME, __timestamp))
    DATA_ROOT_DIR_PATH=Path(os.path.join(ARITFACTS_ROOT_DIR_PATH, DataTransformationConstants.DATA_ROOT_DIR_NAME))
    TRANSFORMATION_ROOT_DIR_PATH=Path(os.path.join(DATA_ROOT_DIR_PATH, DataTransformationConstants.TRANSFORMATION_ROOT_DIR_NAME))
    TRAIN_DATA_DIR_PATH=Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TRAIN_DATA_DIR_NAME))
    VALIDATION_DATA_DIR_PATH=Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.VALIDATION_DATA_DIR_NAME))
    TEST_DATA_DIR_PATH=Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TEST_DATA_DIR_NAME))
    TOKENIZER_PATH=Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TOKENIZER_NAME))
    MODEL_REPO_ID=DataTransformationConstants.MODEL_REPO_ID



@dataclass(frozen=True)
class ModelTrainerConfig:
    __timestamp = timestamp
    ARITFACTS_ROOT_DIR_PATH = Path(os.path.join(ModelTrainerConstants.ARITFACTS_ROOT_DIR_NAME, __timestamp))
    MODEL_ROOT_DIR_PATH = Path(os.path.join(ARITFACTS_ROOT_DIR_PATH, ModelTrainerConstants.MODEL_ROOT_DIR_NAME))
    TRAINER_ROOT_DIR_PATH = Path(os.path.join(MODEL_ROOT_DIR_PATH, ModelTrainerConstants.TRAINER_ROOT_DIR_NAME))
    BASE_ESTIMATOR_PATH = Path(os.path.join(TRAINER_ROOT_DIR_PATH, ModelTrainerConstants.BASE_ESTIMATOR_NAME))
    FINETUNED_ESTIMATOR_PATH = Path(os.path.join(TRAINER_ROOT_DIR_PATH, ModelTrainerConstants.FINETUNED_ESTIMATOR_NAME))
    PARAMS_FILE_PATH = Path(ModelTrainerConstants.PARAMS_FILE_NAME)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    __timestamp = timestamp
    ARITFACTS_ROOT_DIR_PATH = Path(os.path.join(ModelEvaluationConstants.ARITFACTS_ROOT_DIR_NAME, __timestamp))
    MODEL_ROOT_DIR_PATH = Path(os.path.join(ARITFACTS_ROOT_DIR_PATH, ModelEvaluationConstants.MODEL_ROOT_DIR_NAME))
    EVALUATION_ROOT_DIR_PATH = Path(os.path.join(MODEL_ROOT_DIR_PATH, ModelEvaluationConstants.EVALUATION_ROOT_DIR_NAME))
    REPORT_FILE_PATH = Path(os.path.join(EVALUATION_ROOT_DIR_PATH, ModelEvaluationConstants.REPORT_FILE_NAME))
    PREDICTION_FILE_PATH = Path(os.path.join(EVALUATION_ROOT_DIR_PATH, f"{__timestamp}_{ModelEvaluationConstants.PREDICTION_FILE_NAME}"))


