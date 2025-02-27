# Text-Summarization

## Workflows

1. Update config.yaml
2. Update .env[optional]
3. Update the constants
4. Update the entity
5. Update the configuration
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml

## Setup github secrets:
1. AWS_ACCESS_KEY_ID
2. AWS_SECRET_ACCESS_KEY
3. AWS_REGION
4. AWS_ECR_LOGIN_URI
5. ECR_REPOSITORY_NAME

## Docker Setup In EC2 commands to be Executed
### optinal

```
sudo apt-get update -y
```
```
sudo apt-get upgrade -y
```

### required

```
curl -fsSL https://get.docker.com -o get-docker.sh
```
```
sudo sh get-docker.sh
```
```
sudo usermod -aG docker ubuntu
```
```
newgrp docker
```

## add environment variables to Ec2 instance
### create/edit the .env file:
```
nano /home/ubuntu/.env
```
####  paste all variables and press CTRL+X, Y, ENTER

## reconnect with runner
```
cd actions-runner
```

```
./run.sh
```

