from text_summarization.entity import DataTransformationArtifacts, ModelTrainerArtifacts
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from text_summarization.utils import create_dirs, load_json
from text_summarization.exception import CustomException
from text_summarization.logger import logging
from dataclasses import dataclass
import torch, sys, os
from pathlib import Path
import datasets



os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
os.environ["WANDB_DISABLED"] = "True"

@dataclass
class ModelTrainerComponents:
    __data_transformation_config:DataTransformationArtifacts
    __model_trainer_config:ModelTrainerArtifacts

    @staticmethod
    def __get_model(repo_id:str) -> AutoModelForSeq2SeqLM:
        """get model from repo id

        Args:
            repo_id (str): repository id of model

        Returns:
            AutoModelForSeq2SeqLM: model loaded from repo_id
        """
        try:
            logging.info("In __get_model")

            # get device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"device in use {{{device}}}")
            
            # get model from hugging face
            model = AutoModelForSeq2SeqLM.from_pretrained(repo_id).to(device)
            logging.info(f"model collected from {{{repo_id}}}")

            logging.info("Out __get_model")
            return model
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def __get_trainer(model:AutoModelForSeq2SeqLM, tokenizer:AutoTokenizer, data_collator:DataCollatorForSeq2Seq, training_args:TrainingArguments, train_data:datasets.Dataset, validation_data:datasets.Dataset, callbacks:list) -> Trainer:
        try:
            logging.info("In __get_trainer")

            # Initialize the Trainer
            trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=validation_data,
            callbacks=callbacks
            )
            logging.info("{{Trainer}} initialized")

            logging.info("Out __get_trainer")
            return trainer
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


    def start_model_training(self) -> ModelTrainerArtifacts:
        """starts model's training and evaluation

        Returns:
            ModelTrainerArtifacts: path of artifacts created throughout training of model
        """
        try:
            logging.info("In start_model_training")
            # create required dir's
            create_dirs(self.__model_trainer_config.ARITFACTS_ROOT_DIR_PATH)
            create_dirs(self.__model_trainer_config.MODEL_ROOT_DIR_PATH)
            create_dirs(self.__model_trainer_config.TRAINER_ROOT_DIR_PATH)
            logging.info("create required dir's")

            # collect data
            # train_data = datasets.load_from_disk(self.__data_transformation_config.TRAIN_DATA_DIR_PATH)
            # validation_data = datasets.load_from_disk(self.__data_transformation_config.VALIDATION_DATA_DIR_PATH)

            # collect less data for faster training
            train_data = datasets.load_from_disk("less_records_artifacts/train")
            validation_data = datasets.load_from_disk("less_records_artifacts/validation")

            logging.info("train and validation data collected for model training")


            repo_id = self.__data_transformation_config.MODEL_REPO_ID
            # get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            logging.info(f"tokenizer collected from {repo_id}")
            # get model
            model = self.__get_model(repo_id)
            logging.info(f"model collected from {repo_id}")


            # get datacollator
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

            # Set up training arguments
            params = load_json(self.__model_trainer_config.PARAMS_FILE_PATH)
            finetuned_model_path = self.__model_trainer_config.TRAINER_ROOT_DIR_PATH

            training_args = TrainingArguments(
                **params,
                output_dir=finetuned_model_path,
                # If using a GPU cluster, you might want to enable fp16 for faster training
                fp16=True if os.environ.get("USE_FP16", "false").lower() == "true" else False,
            )

            # get trainer 
            trainer = self.__get_trainer(
                model=model, 
                tokenizer=tokenizer,
                data_collator=data_collator,
                training_args=training_args,
                train_data=train_data,
                validation_data=validation_data,
                callbacks=[]
            )

            # start training
            trainer.train()

            logging.info(f"finetuned model saved at {finetuned_model_path}")

            logging.info("Out start_model_training")
            return self.__model_trainer_config
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        

