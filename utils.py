import time
import json
import logging
import logging.config
import tensorflow as tf


class Timer:
    def __init__(self):
        self._tic = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.eclipsed = time.time() - self._tic


DEFAULT_LOGGER = None


def get_default_logger():
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = logging.getLogger('ALL')
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s'))

        DEFAULT_LOGGER.setLevel(logging.DEBUG)
        DEFAULT_LOGGER.addHandler(handler)
    return DEFAULT_LOGGER


##################################################################
#                       file utilities                           #
##################################################################
def delete_if_exists(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)


def load_config(path=None):
    path = 'config.json' if path is None else path
    with open(path, 'r') as f:
        config = json.load(f)
    return config


###################################################################
#                        tf utilities                             #
###################################################################
def huber_loss(x):
    pass


###################################################################
#                        image utilities                          #
###################################################################


