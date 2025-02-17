from dataclasses import dataclass
from text_summarization.components import (
    DataIngestionComponents
)
from text_summarization.configuration import (
    DataIngestionConfig
)


@dataclass
class DataIngestionPipeline:

    def main(self) -> None:
        self.data_ingestion = DataIngestionComponents(DataIngestionConfig)
        self.data_ingestion.start_data_ingestion()





STAGE_NAME = "Data Ingestion"

if __name__=="__main__":
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} initiated <<<<<<<<<<<<<<<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    print(f"\n>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<")


