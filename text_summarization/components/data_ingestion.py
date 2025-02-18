from text_summarization.exception import CustomException
from text_summarization.logger import logging
from text_summarization.utils import (
    create_dirs
)
from text_summarization.entity import (
    DataIngestionArtifacts
)
from dataclasses import dataclass
from urllib.request import urlretrieve
from  zipfile import ZipFile
import sys


@dataclass
class DataIngestionComponents:
    __data_ingestion_config:DataIngestionArtifacts

    @staticmethod
    def __download(source_uri:str, zip_file_path:str) -> None:
        """Description: downloads the data zip file and saves locally

        Args:
            source_uri (str): uri for downloading
            zip_file_path (str): path to save file locally
        """
        try:
            logging.info("Downloading........")
            urlretrieve(source_uri, zip_file_path)
            logging.info("Download complete.")            
        except Exception as e:
            logging.error(e)
            CustomException(e, sys)
    
    @staticmethod
    def __extract(zip_file_path:str, raw_data_dir:str) -> None:
        """Description: extracts the zip file into given path

        Args:
            zip_file_path (str): path of zip file needed to be extracted
            raw_data_dir (str): path of directory for extraction
        """
        try:

            with ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)
                logging.info("zip extraction comleted.")
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
    
    def start_data_ingestion(self)->DataIngestionArtifacts:
        """Runs Data ingestion
        """
        try:
            # create required dir's
            create_dirs(self.__data_ingestion_config.ARITFACTS_ROOT_DIR_PATH)
            create_dirs(self.__data_ingestion_config.DATA_ROOT_DIR_PATH)
            create_dirs(self.__data_ingestion_config.INGESTION_ROOT_DIR_PATH)
            create_dirs(self.__data_ingestion_config.FEATURE_STORE_ROOT_DIR_PATH)
            create_dirs(self.__data_ingestion_config.INGESTED_ROOT_DIR_PATH)

            # get required variables
            uri = self.__data_ingestion_config.SOURCE_URI
            zip_file_path=self.__data_ingestion_config.ZIP_FILE_PATH
            ingested_data_dir = self.__data_ingestion_config.INGESTED_ROOT_DIR_PATH

            # run process
            self.__download(uri, zip_file_path)
            self.__extract(zip_file_path, ingested_data_dir)

            return self.__data_ingestion_config
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        

