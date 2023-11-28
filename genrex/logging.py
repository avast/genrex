import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)-8s] -- %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("GenRex")


logger = setup_logging()
