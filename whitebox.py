"""
Collection of functions and parameters for analysing a Deep Neural Network
(DNN) as a white box.
"""

import numpy as np
from keras import Model
import scipy.optimize
import copy
from blackbox_src.sign_recovery import getHiddenVector,getLocalMatrixAndBias

def getWeightsAndBiases(model, layers):
    weights = []
    biases = []
    for l in layers:
        w, b = model.get_layer(index=l).get_weights()
        weights.append(np.copy(w))
        biases.append(np.copy(b))
    return weights, biases

def getAllWeightsAndBiases_quantized(interpreter):
    weights = []
    biases = []
    for layer in interpreter.get_tensor_details():
        layer_name = layer['name']
        try:
            # Skip layers with specific substrings in their names
            if "MatMul1" in layer_name or "ReadVariableOp1" in layer_name or "StatefulPartitionedCall" in layer_name:
                continue
            tensor_index = layer['index']
            tensor = interpreter.tensor(tensor_index)()
            if "MatMul" in layer_name:
                weights.append(np.transpose(np.copy(tensor)))
            if "ReadVariableOp" in layer_name:
                biases.append(np.transpose(np.copy(tensor)))
        except ValueError as e:
            pass
    weights = weights[::-1]
    biases = biases[::-1]
    return weights, biases
def getWeightsAndBiases_quantized(interpreter, layers):
    weights = []
    biases = []
    for layer in interpreter.get_tensor_details():
        layer_name = layer['name']
        try:
            # Skip layers with specific substrings in their names
            if "MatMul1" in layer_name or "ReadVariableOp1" in layer_name or "StatefulPartitionedCall" in layer_name:
                continue
            tensor_index = layer['index']
            tensor = interpreter.tensor(tensor_index)()
            if "MatMul" in layer_name:
                weights.append(np.transpose(np.copy(tensor)))
            if "ReadVariableOp" in layer_name:
                biases.append(np.transpose(np.copy(tensor)))
        except ValueError as e:
            pass
    weights = weights[::-1]
    biases = biases[::-1]
    # Create separate lists for the desired layers
    selected_weights = [weights[i] for i in layers]
    selected_biases = [biases[i] for i in layers]
    
    return selected_weights, selected_biases

def getWhiteboxRealSigns(model, layerID):
    weights, biases = getWeightsAndBiases(model, range(1, layerID + 1))
    signsLayer = np.sign(weights[-1][0])
    #Changed a bit here so a_k possible if a_1=0
    for i in range(len(signsLayer)):
        c = 1
        while signsLayer[i]==0 and c<weights[-1].shape[1]:
            signsLayer[i]=np.sign(weights[-1][c][i])
            c+=1
    return signsLayer

def getRealSigns_quantized(interpreter, layerID):
    weights, biases = getWeightsAndBiases_quantized(interpreter, range(0, layerID))
    signsLayer = np.sign(weights[-1][0])
    #Changed a bit here so a_k possible if a_1=0
    for i in range(len(signsLayer)):
        c = 1
        while signsLayer[i]==0 and c<weights[-1].shape[1]:
            signsLayer[i]=np.sign(weights[-1][c][i])
            c+=1
    return signsLayer

def getWhiteboxSignatures(model, layerID):
    """Simulates the signature recovery and returns the corresponding weights, biases."""
    # Update signs as they would be recovered as signatures
    #Be careful here multiply by sign vector but not obtained solely from a_1 but from different a_k bcs a_1 can be 0 in pruned network
    #when multiplying from signature back to parameters by a_k have to be careful to not multiply by a_1
    #Not quite right because it only multiplies the weights and biases by the sign, whereas they are supposed to already be the correct weights and biases?
    weights, biases = getWeightsAndBiases(model, range(1, layerID + 1))

    signsLayer = getWhiteboxRealSigns(model,layerID)
    weights[-1] = signsLayer[np.newaxis, :] * weights[-1]
    biases[-1] = signsLayer * biases[-1]
    return weights, biases
def getSignatures_quantized(interpreter, layerID):
    """Simulates the signature recovery and returns the corresponding weights, biases."""
    weights, biases = getWeightsAndBiases_quantized(interpreter, range(0, layerID))
    signsLayer = getRealSigns_quantized(interpreter,layerID)
    weights[-1] = signsLayer[np.newaxis, :] * weights[-1]
    biases[-1] = signsLayer * biases[-1]
    return weights, biases

def signIsCorrect(neuronID, w, w0):
    return (w[:,neuronID]==w0[:,neuronID]).all()

def getScrambledSigns(w, b):
    w = w.copy()
    b = b.copy()
    nNeurons = w.shape[-1]
    #------------------------------
    # my sign guess and starting point
    #------------------------------
    # as a starting point, we assume that half of the signatures have wrong signs
    for nID in range(nNeurons):
        sign = np.random.choice([+1, -1])
        w[:,nID] = sign * w[:,nID]
        b[nID] = sign * b[nID]
    return w, b

def toggleSign(neuronID, w, b): 
    w[:,neuronID] = (-1) * w[:,neuronID]
    b[neuronID] = (-1) * b[neuronID]
    return w, b

def getTogglingPoints(model, layerID, neuronID, funcEps): 
    """Find at which `epsilon` values a function of epsilon `funcEps` leads to the toggling of a specific neuron
    `neuronID` in layer `layerID` of a TensorFlow model `model`.
    
    For example: 
    >>> funcEps = lambda x: deti.interpol.linearMorphEps(myfrog, mycar, x)
    >>> getTogglingPoints(model, layerID, neuronID, funcEps)
    """
    import scipy.optimize
    
    weights, bias = getNeuronWeightBias(model, layerID, neuronID)
    func = lambda x: getLiEquation(x, funcEps, weights, bias) # the neuron will be toggled when this equation is equal to zero.
    epsilons = scipy.optimize.fsolve(func, 0) 
    
    return epsilons

def getLiEquation(epsilon, funcMorphEpsilon, weights, bias):
    """Given the neurons `weights` w1...wn and `bias` b, return the equation
    
        w1 * p1 + ... + wn * pn + b,
        
    where the values of `p` are given by a morph function dependent on `epsilon` 
    
        (p1, ..., pn) = funcMorphEpsilon(epsilon).
    """
    pvec = funcMorphEpsilon(epsilon) # morphed image at position epsilon
    pvec = pvec.flatten()            # flattened morphed image
    LiEquation = np.dot(weights, pvec.flatten()) + bias
    return LiEquation

def getNeuronWeightBias(model, layerID, neuronID): 
    """Get the neuron weights and bias of neuron `neuronID` in layer `layerID` of a TensorFlow model.
    """
    
    weightsAndBiases = model.layers[layerID].weights
    
    weights = weightsAndBiases[0]
    weightsOfNeuron = weights.numpy()[:, neuronID]
    
    bias = weightsAndBiases[1]
    biasOfNeuron = bias.numpy()[neuronID]
    
    return weightsOfNeuron, biasOfNeuron

def getNeuronSignature(model, layerID, neuronID): 
    """
    Get the neuron signature of neuron `neuronID` in layer `layerID` of a TensorFlow model. 
    The neuron signature is obtained by dividing the weight of each incoming connection `w1...wn` by the weight of the 
    first connection `w1`, i.e. 
    
        (w1/w1, w2/w1, ..., wn/w1).
    
    To obtain the weights and biases themselves, please use getNeuronWeightBias. 
    """
    
    weightsOfNeuron, _ = getNeuronWeightBias(model, layerID, neuronID)
    #Must change to wk because it does not have to be a_1 but can be a_k for pruned networks
    w1 = weightsOfNeuron[0]
    return [w/w1 for w in weightsOfNeuron]

def getLayerOutputs(model, testInput, onlyLayerID=None):
    """
    For a neural network model, collect the intermediate outputs of all layers*  for a test input.
    
    *or only one particular layer identified by its `layerID` in model.layers via the  `onlyLayerID` parameter
    """
    
    outputOfAllLayers = []

    for layerID, layer in enumerate(model.layers):
        
        if onlyLayerID is not None and layerID != onlyLayerID:
            continue

        intermediateLayerModel = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
        intermediateOutput = intermediateLayerModel.predict(testInput)
        outputOfAllLayers.append(intermediateOutput)
        
    if onlyLayerID is not None: outputOfAllLayers = outputOfAllLayers[0]

    return outputOfAllLayers

def findToggledNeuronsInLayer(model, layerID, interpolatedImages, debug=False):
    """
    For a given model find the toggled neurons in layer `layer_id` when moving from image x1 to x2
    via the interpolatedImages.
    
    Get the `interpolatedImages` by using (for example) the function `getInterpolatedImages`.
     
    Returns:
        An array that contains in which of the `n` steps which neuron was toggled.
        For example, the following output means that first neuron 12 was toggled in step 3007:
        array([[3007,   12],
               [6103,   19],
               [7742,    4],
               [8067,    2],
               [9543,   15],
               [9556,   15],
               [9557,   15]])
    """
     
    #-----------------------------------------------
    # Get layer outputs for interpolated images
    #-----------------------------------------------
    outputLayer = getLayerOutputs(model, interpolatedImages, onlyLayerID=layerID)
    
    #-----------------------------------------------
    # Analyze activity and toggling
    #----------------------------------------------- 
    activeInLayer = (outputLayer > 0).astype(int)        # find if the neuron was active or not
    toggled = np.diff(activeInLayer, axis=0)             # find toggling points for each neuron (axis=0)
    if debug:
        print(toggled)
    
    #-----------------------------------------------
    # Return which neuron was toggled in which step
    #-----------------------------------------------
    toggledStepNeuron = np.argwhere((toggled == 1) | (toggled == -1))
    return toggledStepNeuron

"""
A simple linear interpolation between two input images x1 and x2
"""
linearMorph = lambda x1, x2, i, steps: x1 + (x2 - x1) / steps * i

def getInterpolatedImages(x1, x2, morph=linearMorph, n=10_000):
    """
    Get the interpolated images between x1 and x2.
    
    morph: morph function to move from x1 to x2. Required functional form is morph(x1, x2, i, n),
    where `n` is the number of steps with which to move x1 into x2 and
    `i=0...n-1` is the current step id.
    """
    #-----------------------------------------------
    # Interpolate between x1, x2
    #-----------------------------------------------
    morphX = np.zeros((n,) + x1.shape)

    for i in range(n):
        morphX[i] = morph(x1, x2, i, n)
     
    return morphX

def collectWeightAndBiasLists(model, layerID):
    """Helper function: Collect lists of all previous layers weight and biases matrices up to (not including) layer `layerID`. 
    
    Returns: Ws, Bs (list of all numpy array weight matrices before layerID, list of all numpy array bias vectors before layerID)
    """
    
    Ws = []
    Bs = []
    
    # for all previous layers, collect the weights and biases: 
    for pID in range(0, layerID):
        
        weightsAndBiases = model.layers[pID].weights
        
        if len(weightsAndBiases) == 0: 
            continue
            
        w = weightsAndBiases[0]
        w = w.numpy()
        
        b = weightsAndBiases[1].numpy()
        
        Ws += [w] 
        Bs += [b]
        
    return Ws, Bs

def getOutputMatrixWhitebox(x, model, layerId, ReLUInOutFunc=False):
    weights, biases = getWeightsAndBiases(model, range(1, layerId + 1))
    # Output of layer layerId before ReLus
    y = getHiddenVector(weights, biases, layerId, x, relu=False)

    weights, biases = getWeightsAndBiases(model, range(layerId + 1, len(model.layers)))
    if ReLUInOutFunc:
        weights[0][y < 0] = 0
    else:
        y[y < 0] = 0
    o, b = getLocalMatrixAndBias(weights, biases, y)
def getAllWeightsAndBiases(model):
    weights = []
    biases = []
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            weights.append(np.copy(w))
            biases.append(np.copy(b))
    return weights, biases
def alignment(model):
    # A1 real weight matrix
    # A2 extracted weight matrix
    # Checks whether the solution is correct
    # Later add for multiple layers and biases!!!
    weights,biases = getAllWeightsAndBiases(model)
    A1 = copy.deepcopy(weights)
    A2 = copy.deepcopy(weights)
    """A1 = [np.array([[ 0.20653395, -0.00874031,  0.31234977,  0.7198933 , -0.07812267],
              [-0.04922444,  0.634213  ,  0.31703934, -0.23813964,  0.23246014],
              [-0.22176811, -0.25437486,  0.02336364, -0.7925965 , -0.7180967 ],
              [-0.17933774, -0.4420981 ,  0.12449854, -0.35833794, -0.5919736 ],
              [ 0.59245914, -0.0303444 ,  0.05698083, -0.650722  , -0.17858113],
              [-0.00284689, -0.44788986,  0.12036221, -0.23778027, -0.10777619],
              [-0.2208317 ,  0.77337134,  0.02326627, -0.52206534,  0.35089377],
              [-0.4801646 ,  0.14540973, -0.80435526, -0.541207  ,  0.14011773],
              [ 0.38552868,  0.09367298, -0.08718948, -0.06030231, -0.62854606],
              [-0.28511667, -0.25637203,  0.42015642,  0.20455015, -0.7419736 ]]),
        np.array([[ 0.17466213, -0.22625984, -0.22964358,  0.30016485,  0.40023768],
              [ 0.38974223, -0.41513768, -0.08056752,  0.12048816,  0.36570254],
              [-0.15090325, -0.12735645, -0.4173901 , -0.5254079 ,  0.28062603],
              [ 0.54504806, -0.07659733,  0.51481193,  0.11308772, -0.24260868],
              [ 0.09519978,  0.63778627,  0.05488183,  0.64587694, -1.1646423 ]]),
        np.array([[ 0.77543104],
              [ 0.01839022],
              [-0.3771324 ],
              [ 0.03486995],
              [-1.903348  ]])]
    A2 = [np.array([[  3.44429009,  -1.26259297,  10.41078911,   2.44899874, -0.12550219],
       [  3.49600216,   3.75694466,  -3.44387383,  -0.58368409, 9.10667415],
       [  0.25763157, -11.60564322, -11.46219235,  -2.62963942, -3.65257269],
       [  1.37284906,  -9.56728346,  -5.18213023,  -2.12651669, -6.34809381],
       [  0.62832929,  -2.88616978,  -9.41046403,   7.0251485 , -0.4357157 ],
       [  1.32723761,  -1.74184356,  -3.43867675,  -0.03375729, -6.4312577 ],
       [  0.25655779,   5.6710301 ,  -7.549886  ,  -2.61853591, 11.10485137],
       [ -8.86964922,   2.26453693,  -7.82670471,  -5.69360365, 2.08794067],
       [ -0.96144098, -10.15835552,  -0.87206619,   4.57144816, 1.34505175],
       [  4.63307723, -11.99153394,   2.95811686,  -3.38080185,-3.68125013]]), 
       np.array([[-0.18714503, -0.5332973 , -2.51700288,  0.24191376, -0.15654683],
       [ 0.63944728,  0.04784406,  2.11110154, -0.6850108 ,  0.06738355],
       [-0.08582504,  0.50155532,  0.41309084, -0.15947109,  0.43114513],
       [-0.30919058, -0.27286195,  1.33723695,  0.32085737,  0.16850231],
       [-0.46847169, -0.0790535 ,  0.44326693,  0.24209981,  0.31049671]]), 
       np.array([[ 0.00113494],
       [-0.02676752],
       [ 0.00066009],
       [-0.20022903],
       [ 0.06778584]])]"""
    """A2[0] = np.array([[ 1.        ,  0.12789857,  1.02498267, -0.71681539,  0.09688215],
       [-2.036822  , -0.1948661 ,  1.        ,  1.        , -0.17935702],
       [-2.42216053,  1.        , -0.95247438, -0.33189835, -0.13150203],
       [-3.06852779, -0.3652488 , -0.1329267 , -0.41098787,  1.        ],
       [ 3.19215567,  0.53739829, -0.48386021, -0.23970446,  0.19359871]])"""#Yes working!!
    """A2[0] = np.array([[ 0.31326793,  0.04006652,  0.3210942 , -0.22455527,  0.03035007],
       [-0.63807101, -0.0610453 ,  0.31326793,  0.31326793, -0.0561868 ],
       [-0.75878522,  0.31326793, -0.29837968, -0.10397311, -0.04119537],
       [-0.96127135, -0.11442074, -0.04164167, -0.12874932,  0.31326793],
       [ 1.        ,  0.16834965, -0.15157789, -0.07509172,  0.06064827]])"""
    """A2[1] = np.array([[-1.14838806e+00, -3.26311886e-01, -9.34917205e+01,  1.00000000e+00, -3.59954168e-01],
       [-3.41031616e-01, -1.24689679e+00, -5.25718142e+02,  4.67108792e-01,  3.22951781e-01],
       [ 1.00000000e+00,  1.00000000e+00,  1.90229806e+02, -1.98793854e-01,  6.95444308e-01],
       [-4.46102062e-01,  2.45658655e-01,  1.00000000e+00, -9.10379364e-01, -9.04390054e-02],
       [-1.30353342e-01, -1.62791447e-01,  1.82479172e+02,  4.79314827e-01,  1.00000000e+00]])
    #A2[1].T[0] = [-0.08523414, -0.4792847, 0.17342798, 0.00091175,0.16636191]"""
    #A2[1].T[0] = [-0.00347578, -0.00838524, -0.0048555,  -0.00427139, -0.00494764,  0.0011705,
    #0.01910174,  0.0082457,  0.00741298, -0.0007405 ]
    #A2[1].T[0] = [ 0.10709827,  0.38689736, -0.31447375, -0.03946372, -0.00332302,  0.11204614, 0.20744012, -0.29404703, -0.00713476,  0.20267062]
    #A2[1].T[0] = [-0.00722632, -0.02610541, 0.02121872, 0.00266276, 0.00022422, -0.00756017, -0.01399677, 0.01984045, 0.00250156,-0.01884914]
    print("A1: ",A1)
    print("A2: ",A2)
    for layer in range(len(A1)-1):
        M_real = np.copy(A1[layer])
        M_fake = np.copy(A2[layer])

        scores = []
        
        for i in range(M_real.shape[1]):
            vec = M_real[:,i:i+1]
            ratio = np.abs(M_fake/vec)

            scores.append(np.std(A2[layer]/vec,axis=0))

        
        i_s, j_s = scipy.optimize.linear_sum_assignment(scores)
        
        for i,j in zip(i_s, j_s):
            vec = M_real[:,i:i+1]
            ratio = np.abs(M_fake/vec)

            ratio = np.median(ratio[:,j])
            #print("Map from", i, j, ratio)

            gap = np.abs(M_fake[:,j]/ratio - M_real[:,i])
            
            A2[layer][:,j] /= ratio
            A2[layer+1][j,:] *= ratio

        A2[layer] = A2[layer][:,j_s]
        A2[layer+1] = A2[layer+1][j_s,:]

    A2[1] *= np.sign(A2[1][0])
    A2[1] *= np.sign(A1[1][0])
    print("A1: ",A1)
    print("A2: ",A2)


