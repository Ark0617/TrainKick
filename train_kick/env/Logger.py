import logging
import logging.handlers


class Logger:
    def getLogger(name):
        logger = logging.getLogger(name)

        if not logger.handlers:
            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(filename="kick.log")

            logger.setLevel(logging.INFO)
            handler1.setLevel(logging.INFO)
            handler2.setLevel(logging.INFO)

            # "%(asctime)s %(levelname)s [%(threadName)s] %(name)s<%(funcName)s>:%(lineno)s %(message)s"
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s [%(threadName)s] %(name)s:%(lineno)s %(message)s"
            )
            handler1.setFormatter(formatter)
            handler2.setFormatter(formatter)

            logger.addHandler(handler1)
            logger.addHandler(handler2)
        return logger


def test():
    logger = Logger.getLogger("test")
    logger.debug('This is a customer debug message')
    logger.info('This is an customer info message')
    logger.warning('This is a customer warning message')
    logger.error('This is an customer error message')
    logger.critical('This is a customer critical message')


# test()
