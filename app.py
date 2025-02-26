from starlette.responses import RedirectResponse
from fastapi.responses import Response
from text_summarization.logger import logging
from text_summarization.pipeline.prediction_pipeline import PredictionPipeline
from text_summarization.pipeline.training_pipeline import TrainingPipeline
from text_summarization.configuration import PredictionConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_summarization.utils import load_json
from fastapi import FastAPI
import uvicorn, torch, os


# What is Text Summarization?

paths = load_json("paths.json")
if not os.path.exists(paths["TOKENIZER_PATH"]) or not os.path.exists(paths["MODEL_PATH"]):
    training_pipeline = TrainingPipeline()
    training_pipeline.run()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(paths["TOKENIZER_PATH"])
model = AutoModelForSeq2SeqLM.from_pretrained(paths["MODEL_PATH"]).to(device)

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run()

        return Response("Training successful !!")
    except Exception as e:
        logging.exception(e)
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(text:str):
    try:
        output_file_path = PredictionConfig.FILE_PATH

        obj = PredictionPipeline()
        output = obj.predict(text, tokenizer, model, output_file_path)
        return output
    except Exception as e:
        logging.exception(e)
        raise Response(f"Error Occurred! {e}")
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


