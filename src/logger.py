# Customer logger model
import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_file_path = os.path.join("logs", log_file) # file path

# Setting up logger
logging.basicConfig(
    filename=log_file_path,
    format = """[ %(asctime)s ] %(lineno)d: %(levelname)s ->
        %(message)s""",
    level=logging.INFO,
)

