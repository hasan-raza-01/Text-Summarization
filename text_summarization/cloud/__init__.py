import os
from text_summarization.logger import logging


class S3Sync:
    def sync(self,s3_bucket_uri:str, folder:str, cmd:str):
        """push the data from local to cloud

        Args:
            s3_bucket_uri (str): url for bucket present inside AwsS3 bucket
            folder (str): local(PC) folder name to sync with bucket
            cmd (str): 
            - push - push the data present in local to cloud
            - pull - pull the data from cloud and save to local
        """
        if cmd.strip().lower()=="push":
            command = f"aws s3 sync {folder} {s3_bucket_uri} "
            os.system(command)
            logging.info("artifacts pushed to cloud")
        elif cmd.strip().lower()=="pull":
            command = f"aws s3 sync  {s3_bucket_uri} {folder} "
            os.system(command)
            logging.info("artifacts pulled from cloud")
        else:
            mssg = f"improper value {cmd} provide for variable cmd"
            logging.error(mssg)
            raise(mssg)


