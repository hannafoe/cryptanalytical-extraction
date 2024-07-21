import multiprocessing as mp
import os
import logging
import tensorflow as tf
import jax


# Global static variables
MPROC_THREADS = max(mp.cpu_count(),1)
BLOCK_MULTIPLY_FACTOR = 2 #8
DEAD_NEURON_THRESHOLD = 2500 #500 #Changes for how long we do the generalized critical point search and try search for neurons that are hard to find
MIN_SAME_SIZE = 4 #6

# global dynamic variables
class GlobalConfig:
    SAVED_QUERIES = []
    query_count = 0
    crit_query_count = 0
    set_Carlini = False # if this parameter is set we revert back to Carlini's signature extraction
    set_save_queries = False # if this parameter is set we save all queries made, but this costs a lot of memory
    BLOCK_ERROR_TOL = 1e-4


# Setup
def setup_environment():
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Configure JAX for float64
    jax.config.update("jax_enable_x64", True)

    # Configure TensorFlow for float64
    tf.keras.backend.set_floatx('float64')

    # Set logging levels
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('jax').setLevel(logging.ERROR)

def set_global_vars(queries_bool=False, carlini_bool=False, block_error=1e-4):
    if queries_bool:
        GlobalConfig.set_save_queries = True
    if carlini_bool:
        GlobalConfig.set_Carlini = True
        GlobalConfig.BLOCK_ERROR_TOL = 1e-3
    if block_error != 1e-4:
        GlobalConfig.BLOCK_ERROR_TOL = block_error