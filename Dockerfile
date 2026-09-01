FROM python:3.12-slim AS base-container
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get upgrade -y && apt-get clean 

RUN apt install awscli -y

RUN pip install uv 

RUN uv pip install --system -e .

EXPOSE 8080

CMD [ "python", "app.py"] 
