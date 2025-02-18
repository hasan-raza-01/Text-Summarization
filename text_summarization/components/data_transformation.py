from dataclasses import dataclass
from text_summarization.entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts
)
from text_summarization.exception import CustomException
from text_summarization.logger import logging
from text_summarization.utils import create_dirs
from transformers import AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
import torch, os, sys


@dataclass
class DataTranformationComponents:
    __data_ingestion_config:DataIngestionArtifacts
    __data_transformation_config:DataTransformationArtifacts

    @staticmethod
    def __get_tokenizer(repo_id:str, path:Path=None) -> AutoTokenizer:
        """get tokenizer from hugging face using repo id and hugging face token

        Args:
            repo_id (str): repository id of model
            path (Path): Path to save tokenizer locally

        Returns:
            AutoTokenizer: tokenizer for model available in repository
        """
        try:
            logging.info("In __get_tokenizer")
            # get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            logging.info(f"tokenizer pulled from {repo_id}")

            # save tokenizer to local
            tokenizer.save_pretrained(Path(path))
            logging.info(f"tokenizer saved at {path}")

            logging.info("Out __get_tokenizer")
            return tokenizer
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def __transform(record:dict, tokenizer:AutoTokenizer, device:str) -> dict:
        """transforms data through tokenizer

        Args:
            record (dict): input data to transform
            tokenizer (AutoTokenizer): tokenizer to perform transformation

        Returns:
            dict: transformed data
        """
        try:
            # Pre-process input text
            output =  tokenizer(record["dialogue"], truncation=True, padding="longest", return_tensors="pt", max_length=1024).to(device)
            return output
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        

    def start_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("In start_data_transformation")
            # create required dir's
            create_dirs(self.__data_transformation_config.ARITFACTS_ROOT_DIR_PATH)
            create_dirs(self.__data_transformation_config.DATA_ROOT_DIR_PATH)
            create_dirs(self.__data_transformation_config.TRANSFORMATION_ROOT_DIR_PATH)
            logging.info("Dir's creation completed")

            # get device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"device setup to [{device}]")

            # collect required data
            train_data = load_from_disk(os.path.join(self.__data_ingestion_config.INGESTED_ROOT_DIR_PATH, "samsum_dataset/train"))
            logging.info("train data collected")

            validation_data = load_from_disk(os.path.join(self.__data_ingestion_config.INGESTED_ROOT_DIR_PATH, "samsum_dataset/validation"))
            logging.info("validation data collected")

            test_data = load_from_disk(os.path.join(self.__data_ingestion_config.INGESTED_ROOT_DIR_PATH, "samsum_dataset/test"))
            logging.info("test data collected")

            # get tokenizer
            repo_id = self.__data_transformation_config.MODEL_REPO_ID
            tokenizer_path = self.__data_transformation_config.TOKENIZER_PATH
            tokenizer = self.__get_tokenizer(repo_id, tokenizer_path)
            logging.info("tokenizer collected successfully")

            # start transformation
            transformed_train_data = train_data.map(self.__transform, fn_kwargs={"tokenizer":tokenizer, "device":device})
            transformed_validation_data = validation_data.map(self.__transform, fn_kwargs={"tokenizer":tokenizer, "device":device})
            transformed_test_data = test_data.map(self.__transform, fn_kwargs={"tokenizer":tokenizer, "device":device})
            logging.info("transformed data collected")

            # get variables of path for train, validation and test
            train_data_path = self.__data_transformation_config.TRAIN_DATA_DIR_PATH
            validation_data_path = self.__data_transformation_config.VALIDATION_DATA_DIR_PATH
            test_data_path = self.__data_transformation_config.TEST_DATA_DIR_PATH
            logging.info("created path to save transformed data")

            # save the datasets
            transformed_train_data.save_to_disk(train_data_path)
            logging.info(f"transformed train data saved at {train_data_path}")

            transformed_validation_data.save_to_disk(validation_data_path)
            logging.info(f"transformed validation data saved at {train_data_path}")

            transformed_test_data.save_to_disk(test_data_path)
            logging.info(f"transformed test data saved at {train_data_path}")

            logging.info("Out start_data_transformation")
            return self.__data_transformation_config
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        

