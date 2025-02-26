from dataclasses import dataclass
from text_summarization.components.data_ingestion import DataIngestionComponents
from text_summarization.components.data_transformation import DataTranformationComponents
from text_summarization.components.model_trainer import ModelTrainerComponents
from text_summarization.components.model_evaluation import ModelEvaluationComponents
from text_summarization.exception import CustomException
from text_summarization.constants import TrainingPipelineConstants
from text_summarization.entity import ModelEvaluationArtifacts
from text_summarization.cloud import S3Sync
from text_summarization.configuration import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from text_summarization.utils import load_json
import sys, os


@dataclass
class TrainingPipeline:

    ## local artifact is going to s3 bucket    
    def push_to_cloud(self):
        try:
            s3_bucket = TrainingPipelineConstants.AWS_BUCKET_NAME
            bucket_folder = TrainingPipelineConstants.AWS_BUCKET_FOLDER_NAME
            local_artifacts_path = load_json("paths.json")["ARTIFACTS_PATH"]
            time_stamp = local_artifacts_path.split(os.sep)[-1]
            aws_s3_bucket_url = f"s3://{s3_bucket}/{bucket_folder}/{time_stamp}"
            # push artifacts to cloud
            syncer = S3Sync()
            syncer.sync(aws_s3_bucket_url, local_artifacts_path, cmd="push")
        except Exception as e:
            raise CustomException(e,sys)


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
        
        # sync model artifacts to S3 =====> step 5
        self.push_to_cloud()
        
        return model_evaluation_artifacts
    

