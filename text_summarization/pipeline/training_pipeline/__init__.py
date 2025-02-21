from dataclasses import dataclass
from text_summarization.components.data_ingestion import DataIngestionComponents
from text_summarization.components.data_transformation import DataTranformationComponents
from text_summarization.components.model_trainer import ModelTrainerComponents
from text_summarization.components.model_evaluation import ModelEvaluationComponents
from text_summarization.entity import ModelEvaluationArtifacts
from text_summarization.configuration import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


@dataclass
class TrainingPipeline:

    def run(self) -> ModelEvaluationArtifacts :
        # data ingsetion =====> step 1
        data_ingestion = DataIngestionComponents(
            DataIngestionConfig
        )
        data_ingsetion_artifacts = data_ingestion.start_data_ingestion()

        # data transformation =====> step 2
        data_transformation = DataTranformationComponents(
            data_ingsetion_artifacts,
            DataTransformationConfig
        )
        data_transformation_artifacts = data_transformation.start_data_transformation()

        # model trainer =====> step 3
        model_trainer = ModelTrainerComponents(
            data_transformation_artifacts,
            ModelTrainerConfig
        )
        model_trainer_artifacts = model_trainer.start_model_training()

        # model evaluation =====> step 4
        model_evaluation = ModelEvaluationComponents(
            data_transformation_artifacts,
            model_trainer_artifacts,
            ModelEvaluationConfig
        )
        model_evaluation_artifacts = model_evaluation.evaluate()

        return model_evaluation_artifacts
    

