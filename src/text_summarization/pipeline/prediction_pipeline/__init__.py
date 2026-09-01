from text_summarization.exception import CustomException
from text_summarization.logger import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_summarization.utils import  create_dirs, save_json
from dataclasses import dataclass
from transformers import pipeline
from datetime import datetime
import os,sys


@dataclass
class PredictionPipeline:

    def predict(self, text:str, tokenizer:AutoTokenizer, model:AutoModelForSeq2SeqLM, output_file_path:str):
        try:
            time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

            pipe = pipeline("summarization", model=model, tokenizer=tokenizer)

            output = pipe(text, **gen_kwargs)[0]["summary_text"]

            # get directory and file name for output file
            dir_path, file_name = os.path.split(output_file_path)
            
            # create dir for output file
            create_dirs(dir_path)

            # save output file to local
            file_path = os.path.join(dir_path, f"{time_stamp}_{file_name}")
            save_json({"input":text, "output":output}, file_path)

            return output
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    

