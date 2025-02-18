from text_summarization.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from text_summarization.pipeline.stage_02_data_transformation import DataTransformationPipeline


STAGE_NAME = "Data Ingestion"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


STAGE_NAME = "Data Transformation"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


