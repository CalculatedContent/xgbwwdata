import logging

from xgbwwdata import Filters, enable_logging

def test_filters_construct():
    f = Filters()
    assert f.min_rows == 200


def test_enable_logging_adds_handler():
    logger = logging.getLogger("xgbwwdata")
    logger.handlers.clear()
    enable_logging()
    assert logger.handlers
    assert logger.level == logging.INFO
