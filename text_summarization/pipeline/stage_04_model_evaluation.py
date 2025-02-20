from text_summarization.configuration import (
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from text_summarization.components.model_evaluation import ModelEvaluationComponents
from text_summarization.logger import logging
from dataclasses import dataclass


@dataclass
class ModelEvaluationPipeline:

    def main(self) -> None:
        self.model_evaluator = ModelEvaluationComponents(DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)
        self.model_evaluator.evaluate()





STAGE_NAME = "Model Evaluation"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


