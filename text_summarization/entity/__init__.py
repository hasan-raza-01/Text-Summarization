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


