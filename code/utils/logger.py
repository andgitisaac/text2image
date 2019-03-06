import logging
from logging import StreamHandler


class LogWriter(object):
    def __init__(self, level=logging.DEBUG):
        self.level = level
        self.content_format = '%(asctime)s - #[%(levelname)s]# %(name)s (line:%(lineno)d) - %(message)s'
        self.date_format = '%m/%d/%Y %I:%M:%S %p'
        self.formatter = logging.Formatter(fmt=self.content_format,
                                            datefmt=self.date_format)
        self.logger_name = None

    
    def create_logger(self, logger_name=''):
        self.logger_name = logger_name            

        # Create logger
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)

        # Create console handler
        ch = StreamHandler()
        ch.setLevel(self.level)
        ch.setFormatter(self.formatter)

        # Add the handlers to the logger
        logger.addHandler(ch)

        return logger