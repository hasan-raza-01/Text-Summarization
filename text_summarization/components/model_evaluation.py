from text_summarization.entity import DataTransformationArtifacts, ModelTrainerArtifacts
from text_summarization.entity import ModelEvaluationArtifacts
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from text_summarization.exception import CustomException
from text_summarization.utils import save_json, create_dirs
from datasets import load_from_disk, load_metric
from text_summarization.logger import logging
from dataclasses import dataclass
import datasets, torch, sys
from tqdm import tqdm



@dataclass
class ModelEvaluationComponents:
    __data_transformation_config: DataTransformationArtifacts
    __model_trainer_config: ModelTrainerArtifacts
    __model_evaluation_config: ModelEvaluationArtifacts

    @staticmethod
    def generate_batch_sized_chunks(list_of_elements:list, batch_size:int):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements.

        Args:
            list_of_elements (list): single column of input feature
            batch_size (int): batch size

        Yields:
            Generator[list, Any, None]
        """
        try:
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i : i + batch_size]
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    
    def calculate_metric_on_test_ds(self, dataset:datasets.Dataset, metric_object, model:AutoModelForSeq2SeqLM, 
                                    tokenizer:AutoTokenizer, batch_size:int = 16, device:str = "cpu", 
                                    column_text:str = "dialogue", column_summary:str = "summary") -> dict:
        """calculates metrics for provided dataset with provide parameters

        Args:
            dataset (datasets.Dataset): test dataset to calculate model performance
            metric_object (_type_): datasets.load_metric object
            model (AutoModelForSeq2SeqLM): model for prediction
            tokenizer (AutoTokenizer): tokenizer for tokenization
            batch_size (int, optional): batch size to insert number of records at a time. Defaults to 16.
            device (str, optional): device to initialize for performing this operation. Defaults to "cpu".
            column_text (str): name of input features column. Defaults to "dialogue".
            column_summary (str): name of output feature column. Defaults to "summary".

        Returns:
            dict: performance report in form of dictionary
        """
        try:
            article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
            target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total=len(article_batches)):
                
                inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                                padding="max_length", return_tensors="pt")
                
                summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device), 
                                length_penalty=0.8, num_beams=8, max_length=128)
                ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
                
                # Finally, we decode the generated texts, 
                # replace the  token, and add the decoded texts with the references to the metric.
                decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=True) 
                    for s in summaries]      
                
                decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
                
                
                metric_object.add_batch(predictions=decoded_summaries, references=target_batch)
                
            #  Finally compute and return the ROUGE scores.
            score = metric_object.compute()
            return score
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


    def evaluate(self) -> ModelEvaluationArtifacts:
        """Performs evaluation

        Returns:
            ModelEvaluationArtifacts: path of artifacts created while performing evaluation
        """
        try:
            # create required dir's
            create_dirs(self.__model_evaluation_config.ARITFACTS_ROOT_DIR_PATH)
            create_dirs(self.__model_evaluation_config.MODEL_ROOT_DIR_PATH)
            create_dirs(self.__model_evaluation_config.EVALUATION_ROOT_DIR_PATH)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.__data_transformation_config.TOKENIZER_PATH)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.__model_trainer_config.FINETUNED_ESTIMATOR_PATH).to(device)
        
            #loading data 
            # dataset = load_from_disk(self.__data_transformation_config.TEST_DATA_DIR_PATH)
            # loading less record data for faster report
            dataset = load_from_disk("less_records_artifacts/test")

            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    
            rouge_metric = load_metric('rouge')

            score = self.calculate_metric_on_test_ds(
                dataset, rouge_metric, model_pegasus, tokenizer, batch_size = 1, 
                device=device, column_text = 'dialogue', column_summary= 'summary'
            )

            rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

            # save report
            report_path = self.__model_evaluation_config.REPORT_FILE_PATH
            save_json(rouge_dict, report_path)

            return self.__model_evaluation_config
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)     
        

