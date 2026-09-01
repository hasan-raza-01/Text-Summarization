from text_summarization.configuration import (
    DataIngestionConfig,
    DataTransformationConfig
)
from text_summarization.components.data_transformation import DataTranformationComponents
from dataclasses import dataclass
from text_summarization.logger import logging


@dataclass
class DataTransformationPipeline:

    def main(self) -> None:
        self.data_ingestion = DataTranformationComponents(DataIngestionConfig, DataTransformationConfig)
        self.data_ingestion.start_data_transformation()





STAGE_NAME = "Data Transformation"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logging.info(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


