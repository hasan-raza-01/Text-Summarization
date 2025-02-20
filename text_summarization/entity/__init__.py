from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifacts:
    # DIR'S
    ARITFACTS_ROOT_DIR_PATH:Path
    DATA_ROOT_DIR_PATH:Path
    INGESTION_ROOT_DIR_PATH:Path
    FEATURE_STORE_ROOT_DIR_PATH:Path
    INGESTED_ROOT_DIR_PATH:Path
    # FILES
    ZIP_FILE_PATH:str
    # URI'S
    SOURCE_URI:str


@dataclass(frozen=True)
class DataTransformationArtifacts:
    ARITFACTS_ROOT_DIR_PATH:Path
    DATA_ROOT_DIR_PATH:Path
    TRANSFORMATION_ROOT_DIR_PATH:Path
    TRAIN_DATA_DIR_PATH:Path
    VALIDATION_DATA_DIR_PATH:Path
    TEST_DATA_DIR_PATH:Path
    TOKENIZER_PATH:Path
    MODEL_REPO_ID:str


@dataclass(frozen=True)
class ModelTrainerArtifacts:
    ARITFACTS_ROOT_DIR_PATH:Path
    MODEL_ROOT_DIR_PATH:Path
    TRAINER_ROOT_DIR_PATH:Path
    BASE_ESTIMATOR_PATH:Path
    FINETUNED_ESTIMATOR_PATH:Path
    PARAMS_FILE_PATH:Path


@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    ARITFACTS_ROOT_DIR_PATH:Path
    MODEL_ROOT_DIR_PATH:Path
    EVALUATION_ROOT_DIR_PATH:Path
    REPORT_FILE_PATH:Path
    PREDICTION_FILE_PATH:Path



