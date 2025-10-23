import logging
import os 
from datetime import datetime
#this thing does that contain any functions.
#its like a configuration file for logging in python
#to log like what is described, simply import logger and import logging.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#strftime is string format time. It formats datetime object into a string with specific pattern
#creates a format like 04_17_2025_14_31_45.log everytime when we execute the file and avoids overwriting the old logs
logs_path = os.path.join(os.getcwd(), "logs") #creates a new folder called "logs"
#getcwd() returns the current working directory
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



if __name__ == "__main__":
    logging.info('Logging has started')