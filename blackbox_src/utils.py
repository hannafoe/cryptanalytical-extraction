import numpy as np
import jax.numpy as jnp
import random

from .global_vars import GlobalConfig

# ==========
#  Possible failure modes defined
# ==========
class AcceptableFailure(Exception):
    """
    Sometimes things fail for entirely acceptable reasons (e.g., we haven't
    queried enough points to have seen all the hyperplanes, or we get stuck
    in a constant zero region). When that happens we throw an AcceptableFailure
    because life is tough but we should just back out and try again after
    making the appropriate correction.
    """
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class GatherMoreData(AcceptableFailure):
    """
    When gathering witnesses to hyperplanes, sometimes we don't have
    enough and need more witnesses to *this particular neuron*.
    This error says that we should gather more examples of that one.
    """
    def __init__(self, data, **kwargs):
        super(GatherMoreData, self).__init__(data=data, **kwargs)
    def __str__(self):
        return "GatherMoreData!!"
    
# ----------
# Most basic Helper functions
# ----------

def basis(i, N):
    """
    Standard basis vector along dimension i
    """
    a = np.zeros(N, dtype=np.float64) #float64
    a[i] = 1
    return a
def matmul(a,b,c,np=np):
    if c is None:
        c = np.zeros(1)
    return np.dot(a,b)+c
def forward(x, A, B, with_relu=False, np=np):
    for i,(a,b) in enumerate(zip(A,B)):
        x = matmul(x,a,b,np)
        if (i < len(A)-1) or with_relu:
            x = x*(x>0)
    return x
def forward_at(point, A, B, d_matrix):
    if len(A) == 0:
        return d_matrix

    mask_vectors = [layer > 0 for layer in get_hidden_layers(point, A, B)]

    h_matrix = np.array(d_matrix)
    for i,(matrix,mask) in enumerate(zip(A, mask_vectors)):
        h_matrix = matmul(h_matrix, matrix, None) * mask
    
    return h_matrix
def get_hidden_layers(x, A, B, flat=False,np=np): #forward pass through all layers that have been found
    """
    Given the weights and biases up to a certain layer, compute a forward pass
    around the vicinity of an input x.
    Kind of like getLocalMatrixAndBias(weights, biases, x0) in blackbox.py
    Returns the output calculation from each layer
    """
    if len(A) == 0: return []
    region = []
    for i,(a,b) in enumerate(zip(A, B)):
        x = matmul(x,a,b,np) #forward pass
        if np == jnp:
            region.append(x)
        else:
            region.append(np.copy(x))
        if i < len(A)-1: #relu
            x = x*(x>0)
    if flat:
        region = np.concatenate(region,axis=0)
    return region
def get_hidden_at(A,B, known_A, known_B, LAYER, x, prior=True):
    """
    Get the hidden value for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_activation = [y for x in get_hidden_layers(x,A,B) for y in x]
    else:
        which_activation = []
    which_activation += list(matmul(forward(x[np.newaxis,:],A,B, with_relu=True), known_A, known_B)[0])
    return tuple(which_activation)
def get_polytope(x, A, B, flat=False):
    if len(A) == 0: return tuple()
    h = get_hidden_layers(x, A, B)#,flat
    h = np.concatenate(h, axis=0)
    return tuple(np.int32(np.sign(h)))

def get_polytope_at(A,B, known_A, known_B, x, prior=True):
    """
    Get the polytope for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_polytope = get_polytope(x,A,B)
    else:
        which_polytope = tuple()
    hidden = forward(x[np.newaxis,:],A,B,with_relu=True)
    which_polytope += tuple(np.int32(np.sign(matmul(hidden, known_A, known_B)))[0])
    return which_polytope
def get_grad(x, direction, weights, biases, eps=1e-6):
    """
    Finite differences to estimate the gradient.
    Uses just two coordinates---that's sufficient for most of the code.

    Can fail if we're right at a critical point and we get the left and right side.
           /
          X
         /
    -X--/

    """
    x = x[np.newaxis,:]
    a = predict_manual_fast(x-eps*direction,weights,biases)
    b = predict_manual_fast(x,weights,biases)
    g1 = (b-a)/eps
    return g1
def get_second_grad_unsigned(x, direction, weights, biases, eps, eps2):
    """
    Compute the second derivative by computing the first derivative twice.
    """
    grad_value = get_grad(x + direction*eps, direction, weights, biases, eps2)+get_grad(x - direction*eps, -direction, weights, biases, eps2)
    return grad_value[0]
        
def getAllWeightsAndBiases(model):
    weights = []
    biases = []
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            weights.append(np.copy(w))
            biases.append(np.copy(b))
    return weights, biases
def predict_manual(x,model):
    """Manual prediction"""
    weights,biases = getAllWeightsAndBiases(model)
    for i in range(len(weights)):
        x = np.matmul(x, weights[i]) + biases[i]
        if i<len(weights)-1:
            x = x*(x>0)
    return x
def predict_manual_fast(x,weights,biases):
    """Manual prediction faster"""
    GlobalConfig.query_count+=x.shape[0] #1
    #print("x: ",x.shape)
    #assert len(x.shape) == 2
    orig_x = x
    for i in range(len(weights)):
        x = matmul(x,weights[i],biases[i])#np.matmul(x, weights[i]) + biases[i]
        if i<len(weights)-1:
            x = x*(x>0)
    if GlobalConfig.set_save_queries:
        GlobalConfig.SAVED_QUERIES.extend(zip(orig_x,x))
    return x
def get_saved_queries(weights,biases,inputShape):
    print("len saved queries: ",len(GlobalConfig.SAVED_QUERIES))
    if len(GlobalConfig.SAVED_QUERIES)<10:
        points =[]
        for i in range(100):
            #inputShape = model.input_shape[1:]
            offset = np.random.normal(0,1,size=inputShape).flatten()
            direction = np.random.normal(0,1,size=inputShape).flatten()
            x = random.uniform(-1000, 1000)
            points.append((offset+direction*x))
        points = np.array(points)
        predict_manual_fast(points,weights,biases)
    return GlobalConfig.SAVED_QUERIES
def get_query_counts():
    print("Query count: ", GlobalConfig.query_count)
    return GlobalConfig.query_count,GlobalConfig.crit_query_count
def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return exp_x / exp_x.sum()

