from logging import getLogger, config
from json import load

def get_logger(name):
    with open("main/log_config.json") as f:
        config.dictConfig(load(f))

    logger = getLogger(name)
    return logger