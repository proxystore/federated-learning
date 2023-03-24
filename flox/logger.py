import logging

logging.basicConfig(
    level=logging.INFO,
    format='[(%(processName)s) %(asctime)s - %(levelname)s] %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S',
    # filename='/temp/myapp.log',
    # filemode='w'
)
