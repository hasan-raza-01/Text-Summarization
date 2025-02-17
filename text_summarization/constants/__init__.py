from text_summarization.utils import (
    read_yaml
)
from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()
CONFIG = read_yaml("config/config.yaml")

@dataclass(frozen=True)
class DataIngestionConstants:
    # DIR'S
    ARITFACTS_ROOT_DIR_NAME=CONFIG.ARITFACTS_ROOT_DIR_NAME
    DATA_ROOT_DIR_NAME=CONFIG.DATA.ROOT_DIR_NAME
    INGESTION_ROOT_DIR_NAME=CONFIG.DATA.INGESTION.ROOT_DIR_NAME
    FEATURE_STORE_ROOT_DIR_NAME=CONFIG.DATA.INGESTION.FEATURE_STORE.ROOT_DIR_NAME
    INGESTED_ROOT_DIR_NAME=CONFIG.DATA.INGESTION.INGESTED.ROOT_DIR_NAME
    # FILES
    ZIP_FILE_NAME=CONFIG.DATA.INGESTION.FEATURE_STORE.ZIP_FILE_NAME
    # URI'S
    SOURCE_URI=os.getenv("SOURCE_URI")


