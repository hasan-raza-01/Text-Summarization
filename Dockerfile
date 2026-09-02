FROM python:3.14-slim AS base-container
WORKDIR /app
COPY . /app

RUN apt-get update \ 
    && apt-get upgrade -y \
    && apt-get autoremove -y

RUN apt-get install -y curl unzip

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip ./aws

RUN pip install uv 

RUN uv pip install --system -e .

EXPOSE 8080

CMD [ "python", "app.py"] 
