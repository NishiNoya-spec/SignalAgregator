import logging
import os
from time import gmtime, strftime


class MainLogger:

    def __init__(self, subpath: str):
        agg_handler = logging.FileHandler(
            os.path.join(subpath, 'aggregation.log')
        )
        formatter = logging.Formatter(
            "%(asctime)s ----- %(levelname)s ----"
            " %(threadName)s:    %(message)s"
        )
        agg_handler.setFormatter(formatter)
        self.logger = logging.getLogger('agg_logger')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(agg_handler)

    def flog(self, message: str, is_print: bool = True, log_type: str = None):
        if is_print:
            print(f"{strftime('%Y-%m-%d %H:%M:%S', gmtime())}|   {message}")
        if log_type:
            log_function = getattr(self.logger, log_type)
            log_function(message)
