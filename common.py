# ---------------------------------------------------
# Prevent file locking errors
# ---------------------------------------------------
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf

def getFormattedTimestamp():
    from datetime import datetime
    # Format the timestamp
    formatted_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    return formatted_timestamp

def getSavePath(modelname, layerID, nExp, mkdir=True):
    from pathlib import Path
        
    pathName = f"results/model_{modelname}/layerID_{layerID}/nExp_{nExp}/"

    if mkdir:
        Path(pathName).mkdir(parents=True, exist_ok=True)

    return pathName

def parseArguments():

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the energy sign recovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument('--model', type=str,
                        help='The path to a keras.model (https://www.tensorflow.org/tutorials/keras/save_and_load).')
    parser.add_argument('--layerID', type=int,
                        help='The ID of your target layer (as enumerated in model.layers).')
    parser.add_argument('--tgtNeurons', nargs='+', 
                        help="Specific target neuron IDs, e.g. '0 10 240'")
    parser.add_argument('--dataset', type=str, 
                        help="If 'None' a random point will be chosen. If 'mnist' we use a little bit of different randomness also we can use points from mnist dataset if specified in 'setting'.")
    parser.add_argument('--eps', type=int, 
                        help="(Optional) Precision parameter for lastLayer method (wiggle size is 10^-eps). Recommended 3 < eps < 8. (But honestly I have never touched this.)")
    parser.add_argument('--quantized', type=int, 
                        help="(Optional) Whether quantized or not. If float16 then specify this as 1 and if float32 then specify this as 2 if float64 then specify this as 0. Beware that with float64 this will run the precision improvement function which is really not great for mnist models. Preferably choose float32.")
    parser.add_argument('--onlySign', type=bool, 
                        help="(Optional) Whether only the sign should be extracted. (True/False)")
    parser.add_argument('--signRecoveryMethod', type=str, 
                        help="(Optional) Whether NeuronWiggle should be used or Carlini's method should be used. Indicate as 'carlini' or 'neuronWiggle'.")
    parser.add_argument('--seed', type=int, 
                        help="(Optional) What seed to use in the parameter extraction.")
    parser.add_argument('--setting',type=str,
                        help="String that describes experimental setting. Only for mnist if we want to try use dataset points in crit. pt. search. (WithDataPoints0.5/original)")

    # ---- default values
    defaults = {'model': "./deti/modelweights/model_cifar10_256_256_256_256.keras",
                'layerID': 2,
                'tgtNeurons': None,
                'nExp': 15,#15, #15 #100 #200 #30
                'dataset': None, 
                'eps': 8, 
                'quantized': 2,
                'onlySign':False,
                'signRecoveryMethod':'neuronWiggle',
                'seed':42,
                'setting':'original'
                }

    # ---- parse args
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    return args

