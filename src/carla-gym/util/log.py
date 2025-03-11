import logging


class CustomFormatter(logging.Formatter):

    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    grey = "\x1b[38;20m"
    reset = "\x1b[0m"
    _format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + _format + reset,
        logging.INFO: grey + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_custom_logger(name, filename: str = None, level=logging.DEBUG, console_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(console_level)

    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler('logs/' + filename)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    logger = setup_custom_logger('test')
    logger.debug("DEBUG")
    logger.info("INFO")
    logger.warning("WARNING")
    logger.error("ERROR")
    logger.critical("CRITICAL")