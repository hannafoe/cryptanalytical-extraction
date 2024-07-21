# All methods in here are a mesh of Carlini et al.'s and Canales-Martinez et al's code and my adaptations
# Their code can be found:
# Carlini: https://github.com/google-research/cryptanalytic-model-extraction
# Canales-Martinez: https://anonymous.4open.science/r/deti-C405
# Beware that while we use tensorflow models as containers of parameters we do not actually use tensorflow in the extraction
# We use jax for extraction
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
import warnings
import multiprocessing as mp
import signal
import tensorflow as tf
import random
import logging

from whitebox import getWhiteboxRealSigns,getWhiteboxSignatures
from common import parseArguments, getSavePath
from blackbox_src.global_vars import setup_environment, set_global_vars
from blackbox_src.utils import get_saved_queries, forward, getAllWeightsAndBiases, get_query_counts, predict_manual_fast, AcceptableFailure
from blackbox_src.sign_recovery import is_solution, solve_contractive_sign, solve_layer_sign, findCorner, findCorner_quantized, getOrthogonalBasisForInnerLayerSpace, getProjection, getWigglesProjection, isLinear
from blackbox_src.critical_point_search import sweep_for_critical_points
from blackbox_src.signature_recovery import recoverCritPts_Signature, check_quality

# ---------------------------------------------------
# Set up logging
# ---------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Tensorflow settings
# ---------------------------------------------------
warnings.filterwarnings('ignore')
setup_environment()


# Don't necessarily need this code but comment back in if needed:
# Prevent Tensorflow from gobbling the whole GPU memory
#devices = tf.config.list_physical_devices('GPU')
#for device in devices:
#    tf.config.experimental.set_memory_growth(device, True)


def solve_second_from_final_layer(A, B,known_A0, known_B0,weights,biases, dimOfLayer,dimOfInput,layerId, pool):
    '''
    Solve the sign for the second from last layer with Carlini's method.
    '''
    starttime = time.time()
    SAVED_QUERIES = get_saved_queries(weights,biases,dimOfInput)
    ## Recover the second to final layer through brute forcing signs, then least squares
    LAYER = len(weights)-2
    assert LAYER==layerId-1
    filtered_inputs = []
    filtered_outputs = []
    # How many unique points to use. This seems to work. Tweak if needed...
    # (In checking consistency of the final layer signs)
    N = int(len(SAVED_QUERIES)/100) or 1
    print("N: ",N)
    filtered_inputs, filtered_outputs = zip(*SAVED_QUERIES[::N])
    print('Total query count', len(SAVED_QUERIES))
    
    print("Solving on", len(filtered_inputs))
    stoptime = time.time()
    tFindCrt =stoptime-starttime

    starttime=time.time()
    inputs, outputs = np.array(filtered_inputs), np.array(filtered_outputs)
    known_hidden_so_far = forward(inputs,A, B, with_relu=True)

    K = dimOfLayer
    print("K IS", K)
    shuf = list(range(1<<K))[::-1]

    print("Here before start", known_hidden_so_far.shape)

    extra_args_tup = (known_A0, known_B0, LAYER-1, known_hidden_so_far, K, -outputs)
    def shufpp(s):
        for elem in s:
            yield elem, extra_args_tup

    # Brute force all sign assignments...
    all_res = pool.map(is_solution, shufpp(shuf))
    scores = [r[0] for r in all_res]
    solution_attempts = sum([r[1] for r in all_res])
    total_attempts = len(all_res)

    print("Attempts at solution:", (solution_attempts), 'out of', total_attempts)
    
    std = np.std([x[0] for x in scores])
    print('std',std)
    print('median', np.median([x[0] for x in scores]))
    print('min', np.min([x[0] for x in scores]))

    score, recovered_signs, final = min(scores,key=lambda x: x[0])
    print('recover', recovered_signs)
    

    known_A0 *= recovered_signs
    known_B0 *= recovered_signs

    out_A,out_B = A+[known_A0], B+[known_B0]
    stoptime=time.time()
    tSignRec = stoptime-starttime
    extracted_sign = np.sign(out_A[-1][0])
    query_count, crit_query_count = get_query_counts()
    return extracted_sign, out_A,out_B, tFindCrt, tSignRec, query_count, crit_query_count

def solve_final_layer(A,B, model,inputShape, lastLayerShape):
    '''
    Solve the signature and sign directly for the last layer through a system of linear equations.
    '''
    real_weights,real_biases = getAllWeightsAndBiases(model)
    # Critical Point Search
    # If we have already saved queries then we use them, else we need to create some
    starttime = time.time()
    SAVED_QUERIES = get_saved_queries(real_weights,real_biases,inputShape)
    N = int(len(SAVED_QUERIES)/1000) or 1
    ins, outs = zip(*SAVED_QUERIES[::N])
    inputs,outputs = np.array(ins),np.array(outs)
    stoptime = time.time()
    tFindCrt_Signature =stoptime-starttime
    avg_tFindCrt_Signature = tFindCrt_Signature/lastLayerShape
    
    # Actual Signature Recovery
    starttime=time.time()
    hidden = forward(inputs, A,B,with_relu=True)
    hidden = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
    solution = np.linalg.lstsq(hidden, outputs)
    vector = solution[0]
    # Report how we're doing
    check_quality(model,len(model.layers)-2, vector[:-1], vector[-1],lastLayerShape,[], do_fix=True)

    At = A+[vector[:-1]]
    Bt = B+[vector[-1]]
    extracted_sign = np.sign(At[-1][0])
    stoptime=time.time()
    tSignatureRec = stoptime-starttime
    avg_tSignatureRec = tSignatureRec/lastLayerShape
    query_count, crit_query_count = get_query_counts()
    avg_query_count, avg_crit_query_count = query_count/lastLayerShape, crit_query_count/lastLayerShape
    return extracted_sign, At,Bt, avg_tFindCrt_Signature, avg_tSignatureRec, avg_query_count, avg_crit_query_count


def init_worker():
    '''
    Signal interrupt workers.
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def recoverSign_Carlini(model, A,B,extracted_normal,extracted_bias, critical_groups, layerId, dataset, special):
    '''
    Exponential time sign recovery method from Carlini.
    '''
    print("Start solving signs!!")
    print("Known already extracted weights and biases: ", A,B)
    print("Extracted normal and bias for target layer: ",extracted_normal,extracted_bias)
    print("Query counts: ",get_query_counts())

    # initiate multiprocessing
    MPROC_THREADS = max(mp.cpu_count(), 1)
    pool = mp.Pool(processes=MPROC_THREADS // 4, initializer=init_worker)
    
    starttime = time.time()
    weights,biases = getAllWeightsAndBiases(model)
    # Return parameters:
    dimOfInput,dimOfPrevLayer,dimOfLayer = weights[0].shape[0],weights[layerId-1].shape[0],weights[layerId-1].shape[1]
    number_of_layers = len(model.layers)
    tFindCrt = 0
    mask = [1]*len(extracted_bias)
    # New critical point generator
    critical_points = sweep_for_critical_points(model, dimOfInput, 1e1,dataset)

    # Solve for signs
    # layer_num = layerId-1
    # len(sizes) = len(weights)? Check
    if layerId-1 == 0 and weights[0].shape[1] <= dimOfInput:
        extracted_sign,crit_query_count,query_count = solve_contractive_sign(A,B,extracted_normal,extracted_bias,model,dimOfLayer)
        #extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
    elif layerId-1 > 0 and weights[0].shape[1] <= dimOfInput and all(weights[x].shape[1] <= weights[x].shape[0]/2 for x in range(1,len(weights))):
        try:
            extracted_sign,crit_query_count,query_count  = solve_contractive_sign(A,B,extracted_normal,extracted_bias,model,dimOfLayer)
        except AcceptableFailure as e:
            print("Contractive solving failed; fall back to noncontractive method")
            if layerId == number_of_layers-2:
                print("Solve second from final layer")
                return solve_second_from_final_layer(A, B,extracted_normal,extracted_bias,weights,biases, dimOfLayer,layerId, pool)
            
            extracted_sign, _,crit_query_count,query_count,tFindCrt  = solve_layer_sign(pool,A,B,extracted_normal,extracted_bias,critical_points,layerId-1,model,dimOfInput,dimOfPrevLayer,dimOfLayer,special,l1_mask=np.int32(np.sign(mask)))
    else:
        if layerId == number_of_layers-2:
            print("Solve second from final layer")
            return solve_second_from_final_layer(A, B,extracted_normal,extracted_bias,weights,biases, dimOfLayer,dimOfInput, layerId, pool)

        extracted_sign, _,crit_query_count,query_count, tFindCrt  = solve_layer_sign(pool,A,B,extracted_normal,extracted_bias,critical_points,layerId-1,model,dimOfInput,dimOfPrevLayer,dimOfLayer, special,l1_mask=np.int32(np.sign(mask)))
        #sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias, critical_points, layer_num, l1_mask=np.int32(np.sign(mask)))

    print("Extracted Sign", extracted_sign)
    print("Real Sign: ", getWhiteboxRealSigns(model, layerId))
    stoptime = time.time()
    tSignRec = stoptime - starttime

    # Correct signs
    extracted_normal *= extracted_sign
    extracted_bias *= extracted_sign
    extracted_bias = np.array(extracted_bias, dtype=np.float64)

    # Report how we're doing
    extracted_normal,extracted_bias,critical_groups, _ = check_quality(model,layerId-1, extracted_normal, extracted_bias,dimOfLayer,critical_groups, do_fix=True)
    print("extracted_normal,extracted_bias: ",extracted_normal,extracted_bias)
    # Add critical query count and query count later in main function to the one obtained from signature extraction
    A.append(extracted_normal)
    B.append(extracted_bias)
    extracted_sign = np.sign(A[-1][0])
    return extracted_sign, A,B, tFindCrt, tSignRec, query_count, crit_query_count


def recoverSign(model, weights, biases, layerId, neuronId,
                inputShape,
                critical_groups,
                nExp=200,
                dataset=None, 
                EPS_IN=1e-6,
                EPS_LYR=1e-8,
                EPS_ZERO=1e-12,
                LINEARITY_EPS=1e-4,
                LINEARITY_ZERO=1e-7, #1e-8
                LINEARITY_DEBUG=False,
                # CHANGED SAMPLE DIFF ZERO
                SAMPLE_DIFF_ZERO=1e-13):
    '''
    Canales-Martinez Neuron Wiggling Sign Recovery.
    Note: If dataset==None: random input point. If dataset='cifar' use input point from CIFAR10 test data.
    '''

    sampleL = []
    sampleR = []
    
    # record the time needed to find critical points
    tFindCrt = 0.0 
    # record the time needed for the sign recovery
    tSignRec = 0.0
    nExpNeg = 0
    nExpPos = 0
    numQueries = 0 #how many queries to model.predict()
    numCritPtQueries = 0 #how many critical points were calculated
    numPrecQueries = 0 #how many times queried in critical points for precision

    orig_weights,orig_biases = getAllWeightsAndBiases(model)

    # Could also parallelize this instead
    i=0
    sample_diff_cnt = 0
    while True:

        # ==========
        #  Critical point
        # ==========
        #
        starttime = time.time()
        if False: #len(critical_groups)!=0 and i<len(critical_groups[neuronId]): # While reusing these critical points is easier, we cannot check for correctness in the sample distance check with these, so let's not reuse them.
            xi = critical_groups[neuronId][i]
            i+=1
        else:
            numCritPtQueries+=1
            if weights[0].dtype == np.float16:
                xi,numPrecQueries = findCorner_quantized(weights, biases, inputShape, [neuronId], numPrecQueries,targetValue=0,timeout=3, dataset=dataset)
            elif weights[0].dtype == np.float32:
                xi,numPrecQueries = findCorner_quantized(weights, biases, inputShape, [neuronId], numPrecQueries,targetValue=0, dataset=dataset)
            else:
                xi,numPrecQueries = findCorner(weights, biases, inputShape, [neuronId],numPrecQueries, targetValue=0, dataset=dataset)
        
        # Get number of active neurons in each hidden layer
        yi = xi
        active = []
        for lyr in range(layerId - 1):
            yi = np.matmul(yi, weights[lyr]) + biases[lyr]
            yi *= (yi > 0)
            active.append(len(yi[yi > 0]))
        active = np.array(active)

        stoptime = time.time()
        tFindCrt += stoptime - starttime

        # ==========
        #  Energy-maximising wiggle
        # ==========
        starttime = time.time()
        #
        # Get orthogonal basis for the input vector space for the target
        # layer and restrict its dimension to that of the minimum dimension
        # in previous layers
        B, diffs = getOrthogonalBasisForInnerLayerSpace(xi, weights, biases, layerId - 1, EPS_IN)
        # assumes that reducing the basis to the size of the smallest layer will capture the most critical aspects of the network's transformations.
        if layerId > 1:
            B = B[:np.min(active)]

        # Get projection of the neuron's signature onto the space above
        proj = getProjection(weights[-1][:, neuronId], B)
        signaturesProj = np.array([proj * (np.abs(proj) > EPS_ZERO)])
        # Get wiggle
        try:
            wigglesi = getWigglesProjection(weights[-1], signaturesProj, diffs, EPS_IN, EPS_LYR)
        except Exception:
            print("Wiggle computation failed.")
            continue

        # ==========
        #  Check linearity
        # ==========
        def gamma(x):
            return predict_manual_fast(xi + wigglesi[0] * x, orig_weights,orig_biases)[0]
            #return model(np.array([xi + wigglesi[0] * x]), training=False).numpy()[0][0] #if tensorflow

        if not (isLinear(gamma,  0.0, 1.0, eps=LINEARITY_EPS, tol=LINEARITY_ZERO, debug=LINEARITY_DEBUG)) or \
        not (isLinear(gamma, -1.0, 0.0, eps=LINEARITY_EPS, tol=LINEARITY_ZERO, debug=LINEARITY_DEBUG)):
            print("Not is linear")
            continue
        numQueries+=2

        # ==========
        # Evaluate DNN
        # ==========
        #
        
        fx = predict_manual_fast(np.array([xi - wigglesi[0], xi + wigglesi[0], xi]),orig_weights,orig_biases)
        #fx = model.predict(np.array([xi - wigglesi[0], xi + wigglesi[0], xi]),verbose=0) #--> a lot slower because of the calls to model.predict() # if tensorflow
        numQueries+=3

        # ==========
        # Samples
        # ==========
        #
        if wigglesi.dtype == np.float16:
            sL = np.linalg.norm(fx[0] - fx[2])
            sR = np.linalg.norm(fx[1] - fx[2])
            sL = sL.astype(np.float16)
            sR = sR.astype(np.float16)
        else:
            sL = np.linalg.norm(fx[0] - fx[2])
            sR = np.linalg.norm(fx[1] - fx[2])
        # Check that samples are "far" from each other
        # We break off a neuron’s sign recovery if more than 30 iterations of this error are seen consecutively
        if (np.abs(sL - sR) < SAMPLE_DIFF_ZERO) or np.abs(sL - sR)==np.nan:
            print("np.abs(sL - sR) < SAMPLE_DIFF_ZERO: ",np.abs(sL - sR),SAMPLE_DIFF_ZERO)
            sample_diff_cnt+=1
            # Put some code here for failure and returning put catch phrase in code that calls recoverSign
            if sample_diff_cnt>15:
                sampleL = np.array(sampleL)
                sampleR = np.array(sampleR)
                print("Neuron ",neuronId,"'s signature is most likely wrong.")
                print("No valid critical points could be computed for the sign extraction.")
                return 0, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec, numQueries, numCritPtQueries,numPrecQueries
            continue
        # If we have passed the check correctly at least once again we can reset the count because we only count continuous errors.
        sample_diff_cnt=0
        # Collect samples
        sampleL.append(sL)
        sampleR.append(sR)

        #Confidence interval
        if sL>sR:
            nExpNeg+=1
        elif sL<sR:
            nExpPos+=1
        
        stoptime = time.time()
        tSignRec += stoptime - starttime
        print("nExpNeg: ",nExpNeg," nExpPos: ",nExpPos," numQueries: ",numQueries)
        if (len(sampleL) == nExp):
            break


    sampleL = np.array(sampleL)
    sampleR = np.array(sampleR)

    # 4: Number of experiments that decided sign +
    m4 = np.sum((sampleL / sampleR) < 1.0)

    signm4 = (-2.0 * (m4 < (len(sampleL)) / 2) + 1.0) if m4 != ((len(sampleL))/ 2) else 0.0
    
    return signm4, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec, numQueries, numCritPtQueries,numPrecQueries




if __name__=='__main__':
    logger.info("""
    # ---------------------------------------------------
    # Starting extracting weights and biases.
    # ---------------------------------------------------   
    """)
    args = parseArguments()
    logger.info(f"Parsed arguments for signature and sign recovery: \n\t {args}.")

    model = tf.keras.models.load_model(args.model)
    logger.info(f"Model summary:")
    logger.info(model.summary())
    # ---------------------------------------------------
    # Filenames
    # ---------------------------------------------------
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    #In case we want to revert the signature extraction back to Carlini's set it in the next line
    #blackbox.set_global_vars(carlini_bool=True)
    # In case the sign recovery should be carlini's
    if args.signRecoveryMethod == 'carlini':
        args.nExp = -2
        if args.layerID == len(model.layers)-2:
            set_global_vars(queries_bool=True)
    if args.layerID == len(model.layers)-1:
        set_global_vars(queries_bool=True)
    # Output files should be marked with the appropriate settings
    modelname = args.model.split('/')[-1].replace('.keras', '')
    quant_level = ""
    if args.quantized==1:
        quant_level = "_float16"
    elif args.quantized==2:
        quant_level="_float32"
    if args.onlySign==True:
        sign_str = "-onlySign"
    else:
        sign_str = ""
    if args.signRecoveryMethod == 'carlini':
        sign_str += "-carlini"
    savePath = getSavePath(modelname, args.layerID, str(args.nExp)+f"-Seed{args.seed}{quant_level}{sign_str}_{args.setting}", mkdir=True)
    filename_pkl = savePath + 'df.pkl'
    filename_md  = savePath + 'df.md'
    ensure_dir(filename_md)
    logger.info(f"Signature and Sign recovery results will be saved to \n\t {filename_md}")
    

    # ---------------------------------------------------
    # Recover signatures (model or extraction)
    # ---------------------------------------------------
    logger.info("Recovering signatures...")

    # Set seed !!!! Do not set it if you are not testing
    np.random.seed(args.seed)
    random.seed(args.seed)
    inputShape = model.input_shape[1:]
    hiddenLayerIDs = [i for i in np.arange(1, len(model.layers)-1)]
    neuronsHiddenLayers = [model.layers[i].output_shape[-1] for i in hiddenLayerIDs]
    outputs = model.output_shape[-1]
    extracted_signs = []

    if args.onlySign == False:
        # If we want prior layers' correct or incorrect extraction to be reflected we would need to load the previous "_extracted" version here.
        # i.e., extracted_model = tf.keras.models.load_model("models/mnist784_16x8_1_extracted_S15.keras")
        # and then weights, biases = getWhiteboxSignatures(extracted_model, args.layerID)
        #
        weights,biases = getWhiteboxSignatures(model, args.layerID)
        # To try out whether signature extraction works when all previous weights are only up to precision of float32
        #for i in range(len(A)):
        #    weights[i] = np.array(weights[i]).astype(np.float32)
        #    biases[i] = np.array(biases[i]).astype(np.float32)
        if args.layerID<len(hiddenLayerIDs)+1:
            # To try out how signature extraction in the target layer is influenced when a neuron from previous layer is scaled to near zero or a sign is flipped.
            #weights[0].T[0]*=-1#0.01
            #biases[0][0]*=-1#0.01
            extracted_normal, extracted_bias, critical_groups, avg_tFindCrt_Signature, avg_tSignatureRec, tImprovePrec, avg_query_count, avg_crit_query_count = recoverCritPts_Signature(model, weights, biases, args.layerID, args.quantized, args.setting, dataset=args.dataset)
            nNeurons = len(critical_groups)

            # CHEATING FUNCTION: Report how well we're doing and scale with same constant as original to make comparable
            # Check quality is a cheating function, so it can be removed if we don't want to cheat
            extracted_normal,extracted_bias,critical_groups,output = check_quality(model,args.layerID-1, extracted_normal, extracted_bias,weights[args.layerID-1].shape[1],critical_groups, do_fix=True)
            extracted_normal,extracted_bias = np.array(extracted_normal),np.array(extracted_bias).flatten() 
            with open(savePath +'quality_check.txt', 'w') as f:
                f.write(output)
            # Check if real and corrected weights and biases are the same up to sign
            weights_same_up_to_sign = np.allclose(np.abs(weights[-1]), np.abs(extracted_normal))
            biases_same_up_to_sign = np.allclose(np.abs(biases[-1]), np.abs(extracted_bias))
            # Check if the weights and bias are same up to sign in float32
            print("Weights, biases same up to sign in float32 precision?: ",weights_same_up_to_sign,biases_same_up_to_sign)
            weights_float16 = [np.array(w, dtype=np.float16) for w in weights]
            biases_float16 = [np.array(b, dtype=np.float16) for b in biases]
            extracted_normal_float16 = np.array(extracted_normal, dtype=np.float16)
            extracted_bias_float16 = np.array(extracted_bias, dtype=np.float16)
            # Check if the weights and biases are the same up to sign in float16 precision
            weights_same_up_to_sign = np.allclose(np.abs(weights_float16[-1]), np.abs(extracted_normal_float16))
            biases_same_up_to_sign = np.allclose(np.abs(biases_float16[-1]), np.abs(extracted_bias_float16))
            print("Weights, biases same up to sign in float16 precision?:", weights_same_up_to_sign, biases_same_up_to_sign)
            filename_weights = savePath + 'weights.npy'
            filename_bias = savePath + 'bias.npy'
            np.save(filename_weights, extracted_normal)
            np.save(filename_bias, extracted_bias)
            cheating = False
            # CHEATING FUNCTION: So we can continue with sign extraction if things fail in signature extraction we can cheat and continue with the actual target model's weights and bias
            # To check what the results of the signature extraction were one can see the saved numpy results above or read the output on the terminal
            if cheating:
                if weights_same_up_to_sign: # only exchange the layer we extracted with the whitebox layer if it is the same up to some precision
                    weights[-1],biases[-1] = extracted_normal,extracted_bias
            else:# NON CHEATING OPTION -> we normally go with this
                weights[-1],biases[-1] = extracted_normal, extracted_bias
        else:
            # Only if last layer then we can brute force
            print("Solve final layer")
            extracted_signs, weights,biases, avg_tFindCrt_Signature, avg_tSignatureRec, avg_query_count, avg_crit_query_count = solve_final_layer(weights[:-1],biases[:-1], model, inputShape, outputs)
            tImprovePrec = 0.0 
    else:
        weights, biases = getWhiteboxSignatures(model, args.layerID)
        critical_groups = []
        avg_tFindCrt_Signature, avg_tSignatureRec, tImprovePrec, avg_query_count, avg_crit_query_count = 0.0,0.0,0.0,0,0

    if args.quantized ==1: #quantized to float16
        print("Signature Extraction to float32 standards.")
        print("Sign Extraction Quantized to float16.")
        # Quantized here means float16 but operations in float16 are slower 
        # because they often have to be converted into float32 for numpy functions
        # and CPU is optimized for float32 and float64
        for i in range(len(weights)):
            weights[i] = np.array(weights[i]).astype(np.float16)
            biases[i] = np.array(biases[i]).astype(np.float16)
    if args.quantized ==2: #quantized to float32
        print("Signature Extraction to float32 standards.")
        print("Sign Extraction Quantized to float32.")
        for i in range(len(weights)):
            weights[i] = np.array(weights[i]).astype(np.float32)
            biases[i] = np.array(biases[i]).astype(np.float32)
    
    # ---------------------------------------------------
    # Inferred settings
    # ---------------------------------------------------
    
    # check output activation function is linear
    original_activation = model.layers[-1].activation
    if model.layers[-1].activation != tf.keras.activations.linear:
        logger.warning(f"The last layer has to have a linear activation function, instead found {model.layers[-1].activation}. We will replace this output function with a linear one automatically in your model.")
        model.layers[-1].activation = tf.keras.activations.linear
    logger.info(f"""
        Determined the following model parameters: 
            input shape: \t {inputShape}
            hiddenLayerIDs: \t {hiddenLayerIDs}
            neuronsHiddenLayers: \t {neuronsHiddenLayers}
            outputs: \t {outputs}
        """)

    # Number of neurons in target layer
    nNeurons = len(biases[-1])

    # Target all neurons if None is specified
    if args.tgtNeurons is None:
        args.tgtNeurons = np.array(range(nNeurons))
    else: 
        # only works if args.onlySign==True because signature recovery recovers for all neurons
        args.tgtNeurons = [int(value) for value in args.tgtNeurons]
    logger.info(f"Signs will be recovered for neuronIDs: \n\t {args.tgtNeurons}.")



    # ---------------------------------------------------
    # Run sign recovery
    # ---------------------------------------------------
    expNeg = []
    expPos = []
    rows = []
    avg_runtime = 0
    

    # WHITEBOX: Get the real signs to be able to control our results:
    whiteSignsLayer = getWhiteboxRealSigns(model, args.layerID)

    logger.info("""
    # NEURON-BY-NEURON SIGN RECOVERY (parallelizable)
    # ---------------------------------------------------""")
    lastLayer = False
    if len(extracted_signs)!=0:
        print("We are done, since in last layer signature recovery, sign recovery happens at the same time!!")
        lastLayer=True
    #args.signRecoveryMethod = "carlini" or "neuronWiggle"
    if args.signRecoveryMethod != 'carlini' and not lastLayer: # NeuronWiggle Sign recovery
        failure_cnt = 0
        for neuronId in args.tgtNeurons:
            # start timer
            starttime = time.time()
            # -------- run the actual sign recovery --------
            # For float16 and float32 we need to change the epsilon
            # float16
            if args.quantized ==1:
                signm4, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec, numQueries, numCritPtQueries,numPrecQueries = recoverSign(model, weights, biases,
                                                                                        args.layerID,
                                                                                        neuronId,
                                                                                        inputShape,
                                                                                        critical_groups,
                                                                                        nExp = args.nExp, 
                                                                                        dataset=args.dataset,                
                                                                                        EPS_IN=2e-2, 
                                                                                        EPS_LYR=2e-2,
                                                                                        EPS_ZERO=1e-3, 
                                                                                        LINEARITY_EPS=6e-1,#1e-1
                                                                                        LINEARITY_ZERO=6e-2,#1e-2
                                                                                        LINEARITY_DEBUG=False,
                                                                                        # CHANGED SAMPLE DIFF ZERO
                                                                                        SAMPLE_DIFF_ZERO=1e-4) #1e-13
            elif args.quantized == 2: #float32
                signm4, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec, numQueries, numCritPtQueries,numPrecQueries = recoverSign(model, weights, biases,
                                                                                        args.layerID,
                                                                                        neuronId,
                                                                                        inputShape,
                                                                                        critical_groups,
                                                                                        nExp = args.nExp, 
                                                                                        dataset=args.dataset,                
                                                                                        EPS_IN=1e-2,#1e-2,#2e-2, #1e-6
                                                                                        EPS_LYR=1e-4,#1e-3,#2e-2, #1e-8
                                                                                        EPS_ZERO=1e-6,#1e-6,#1e-3,  #1e-12
                                                                                        LINEARITY_EPS=1e-1,#1e-2,#1e-1, #1e-4
                                                                                        LINEARITY_ZERO=1e-2,#1e-4,#1e-2, #1e-8
                                                                                        LINEARITY_DEBUG=False,
                                                                                        # CHANGED SAMPLE DIFF ZERO
                                                                                        SAMPLE_DIFF_ZERO=1e-9) 
            else: #float64
                signm4, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec, numQueries, numCritPtQueries,numPrecQueries = recoverSign(model, weights, biases,
                                                                                        args.layerID,
                                                                                        neuronId,
                                                                                        inputShape,
                                                                                        critical_groups,
                                                                                        nExp = args.nExp, 
                                                                                        dataset=args.dataset)

            # stop timer
            stoptime = time.time()

            nExpMax = max(nExpNeg, nExpPos)
            expNeg.append(nExpNeg)
            expPos.append(nExpPos)
            signm4 = signm4*np.sign(weights[-1][0][neuronId])
            extracted_signs.append(signm4)

            # ---------------------------------------------------
            # Load whitebox information to check the sign recover
            # ---------------------------------------------------
            whiteIsCorrect = signm4 == whiteSignsLayer[neuronId]
            whiteResult = "OK" if whiteIsCorrect else "NO <====== Failure!"
            whiteRealSign = '+' if (whiteSignsLayer[neuronId] > 0) else '-'
            if signm4==0:
                whiteRealSign = 0

            # ---------------------------------------------------
            # Log results
            # ---------------------------------------------------
            
            runtime = stoptime-starttime+avg_tSignatureRec+tImprovePrec
            avg_runtime += runtime
            my_ratio = nExpMax / len(sampleL) if len(sampleL)!=0 else 0
            my_percentage = nExpMax / (nExpPos+nExpNeg) if (nExpPos+nExpNeg)!=0 else 0
            logger.info(f"NeuronID: {neuronId} \t -:{nExpNeg}, +:{nExpPos}, \t ratio ({my_ratio:.2f}) \t runtime:{runtime:.2f} seconds \t White-box evaluation: real sign {whiteRealSign} ==> sign recovery={whiteResult}")
            rows.append({'modelID': modelname,
                        'layerID': args.layerID,
                        'neuronID': neuronId,
                        'realSign': whiteRealSign,
                        'metric4Minus': nExpNeg,
                        'metric4Plus': nExpPos,
                        'percentage': my_percentage,#args.nExp,
                        'isRecoveredCorrectly': whiteIsCorrect,
                        'tFindCrit': tFindCrt+avg_tFindCrt_Signature, 
                        'tSignRec': tSignRec,
                        'tSignatureRec': avg_tSignatureRec,
                        'tImprovePrec':tImprovePrec,
                        'recoveryTimeSeconds': runtime,
                        'numQueries': numQueries+avg_query_count,
                        'numCritPtQueries': numCritPtQueries+avg_crit_query_count,
                        'numPrecQueries': numPrecQueries,
                        })

            logger.debug(f"Saving results to {filename_md} and {filename_pkl}...")
            df = pd.DataFrame(rows)
            try:
                df.to_pickle(filename_pkl)
                df.to_markdown(filename_md)
            except OSError as e:
                print(f"An error occurred while saving the dataframe: {e}")
            filename_np = savePath + f"neuronID_{neuronId}_samples.npz"
            logger.debug(f"Saving sign evaluations to {filename_np}...")
            # load using
            # ... data = np.load(..)
            # ... data["samplesL"]
            data = {"samplesL": sampleL,
                    "samplesR": sampleR}
            np.savez(filename_np, **data)
            if signm4==0:
                failure_cnt +=1
            if failure_cnt>5:
                raise RuntimeError("The probability that the whole signature extraction was wrong is high. Probably the previous layer's sign extraction was wrong. Try with a different combination of signs.")
        expNeg = np.array(expNeg)
        expPos = np.array(expPos)
        logger.info(f"Average run time: {avg_runtime/len(args.tgtNeurons):.2f}")
        if failure_cnt<=5 and failure_cnt>0:
            print("Some of the signature extractions must have either been very imprecise or wrong.")
            print("Signature extraction was wrong for at least",failure_cnt, " neurons.")
            print("If this is being run on high precision perhaps switch to lower precision in sign extraction with the option --quantized.")
            print("'--quantized 1' will sign extract at float16 precision and '--quantized 12' will extract at float32 precision.")
        model_name_split = args.model.split('.')
        model_save_path = model_name_split[0]+f"_extracted_S{args.nExp}{quant_level}."+model_name_split[1]
    else: #Carlini's sign recovery
        if lastLayer==True:
            avg_tFindCrt_Sign, avg_tSignRec = 0.0,0.0
            query_count, crit_query_count = get_query_counts()
        elif args.layerID==len(hiddenLayerIDs)+1: # This is in case we accidentally set that we want only sign extraction in last layer
            print("Solve final layer")
            A,B = getWhiteboxSignatures(model, args.layerID)
            extracted_signs, weights, biases, avg_tFindCrt_Sign, avg_tSignRec, _, _ = solve_final_layer(A[:-1],B[:-1], model, inputShape, outputs)
            query_count, crit_query_count = get_query_counts()
        else: # This is Carlini's sign extraction
            extracted_signs, weights, biases, tFindCrt, tSignRec, query_count, crit_query_count = recoverSign_Carlini(model, weights[:-1], biases[:-1],weights[-1],biases[-1], critical_groups,args.layerID, args.dataset,args.quantized)
            avg_tFindCrt_Sign = tFindCrt/nNeurons
            avg_tSignRec = tSignRec/nNeurons
        print("Number of queries average: ",avg_query_count,query_count)
        # ---------------------------------------------------
        # Load whitebox information to check the sign recover
        # ---------------------------------------------------
        print("Extracted signs: ",extracted_signs, " Correct Sign: ",whiteSignsLayer)
        for neuronId in range(nNeurons):
            whiteIsCorrect = extracted_signs[neuronId] == whiteSignsLayer[neuronId]
            whiteResult = "OK" if whiteIsCorrect else "NO <====== Failure!"
            whiteRealSign = '+' if (whiteSignsLayer[neuronId] > 0) else '-'
            # ---------------------------------------------------
            # Log results
            # ---------------------------------------------------
            
            runtime = avg_tSignRec+avg_tSignatureRec+tImprovePrec
            avg_runtime += runtime
            logger.info(f"NeuronID: {neuronId} \t runtime:{runtime:.2f} seconds \t White-box evaluation: real sign {whiteRealSign} ==> sign recovery={whiteResult}")
            rows.append({'modelID': modelname,
                        'layerID': args.layerID,
                        'neuronID': neuronId,
                        'realSign': whiteRealSign,
                        'isRecoveredCorrectly': whiteIsCorrect,
                        'tFindCrit': avg_tFindCrt_Sign+avg_tFindCrt_Signature, 
                        'tSignRec': avg_tSignRec,
                        'tSignatureRec': avg_tSignatureRec,
                        'tImprovePrec':tImprovePrec,
                        'recoveryTimeSeconds': runtime,
                        'numQueries': query_count/nNeurons,
                        'numCritPtQueries': crit_query_count/nNeurons,
                        })
            logger.debug(f"Saving results to {filename_md} and {filename_pkl}...")
            df = pd.DataFrame(rows)
            df.to_pickle(filename_pkl)
            df.to_markdown(filename_md)

            filename_np = savePath + f"neuronID_{neuronId}_samples.npz"
            logger.debug(f"Saving evaluations to {filename_np}...")
        logger.info(f"Average run time: {avg_runtime/nNeurons:.2f}")
        model_name_split = args.model.split('.')
        model_save_path = model_name_split[0]+f"_extracted_CarliniSign{quant_level}."+model_name_split[1]
    if args.layerID==len(hiddenLayerIDs)+1:
        layer = model.get_layer(name=f"output")
    else:
        layer = model.get_layer(name=f"layer{args.layerID-1}")
    weight_matrix_layer, _ = layer.get_weights()
    weight_matrix_layer = np.array(weight_matrix_layer)
    if len(args.tgtNeurons)==len(weight_matrix_layer[0]): #If we have extracted sign of all neurons then save extracted model
        original_weight, original_bias = layer.get_weights()
        extracted_weight,extracted_bias = weights[-1].copy(),biases[-1].copy()
        indices_diff = []
        for i in range(len(original_weight.T)):
            if np.sign(extracted_weight[0][i]) != np.sign(extracted_signs[i]):
                extracted_weight.T[i]*=-1
                extracted_bias[i]*=-1
            weights_equal_exact = np.array_equal(np.sign(original_weight.T[i]), np.sign(extracted_weight.T[i]))
            biases_equal_exact = np.sign(original_bias[i]) == np.sign(extracted_bias[i])
            print("Equal weights and biases: ",weights_equal_exact,biases_equal_exact)
            if not weights_equal_exact:
                indices_diff.append(i)
        layer.set_weights([extracted_weight,extracted_bias])
        # Set output activation to original activation
        model.layers[-1].activation = original_activation
        original_weight, original_bias = layer.get_weights()
        print("Different indices: ", indices_diff)
        # Save the newly extracted model as a tensorflow model
        model.save(model_save_path)
