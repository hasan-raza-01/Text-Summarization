from text_summarization.configuration import (
    DataTransformationConfig,
    ModelTrainerConfig
)
from text_summarization.components.model_trainer import ModelTrainerComponents
from text_summarization.logger import logging
from dataclasses import dataclass


@dataclass
class ModelTrainerPipeline:

    def main(self) -> None:
        self.model_trainer = ModelTrainerComponents(DataTransformationConfig, ModelTrainerConfig)
        self.model_trainer.start_model_training()





STAGE_NAME = "Model Training"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


