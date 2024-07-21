
import numpy as np
import random
import gc
import time
import sys
from io import StringIO
import jax
import jax.numpy as jnp
import networkx as nx
from .global_vars import GlobalConfig,MIN_SAME_SIZE,BLOCK_MULTIPLY_FACTOR, DEAD_NEURON_THRESHOLD
from .utils import get_query_counts, get_hidden_layers, get_hidden_at, forward,forward_at, get_polytope_at, matmul, get_second_grad_unsigned,basis, AcceptableFailure, GatherMoreData
from .critical_point_search import sweep_for_critical_points,do_better_sweep, getAllWeightsAndBiases
from .precision_improvement import improve_layer_precision
# ==========
#  Functions for Signature Recovery
# ==========
# Main function for signature recovery
def recoverCritPts_Signature(model, weights,biases,layerId,
                special,setting,
                dataset=None, ):
    '''
    This is the function for signature recovery adapted from Carlini.
    '''

    # Return parameters:
    PARAM_SEARCH_AT_LOCATION = 1e2
    if len(model.layers) == 3:
        PARAM_SEARCH_AT_LOCATION = 1e4
    dimOfInput,dimOfPrevLayer,dimOfLayer = weights[0].shape[0],weights[layerId-1].shape[0],weights[layerId-1].shape[1]
    
    # First setup the critical points generator
    starttime = time.time()
    critical_points_yielder = sweep_for_critical_points(model, dimOfInput, PARAM_SEARCH_AT_LOCATION,dataset,setting)
    stoptime = time.time()
    tFindCrt = stoptime-starttime

    # Extract weights corresponding to those critical points
    starttime = time.time()
    extracted_normal, extracted_bias, critical_groups, tCritExtra = compute_layer_values(critical_points_yielder,model,weights,biases,layerId,dimOfInput,dimOfPrevLayer,dimOfLayer,special)
    tFindCrt += tCritExtra
    avg_tFindCrt = tFindCrt/len(critical_groups)
    stoptime = time.time()
    tSignatureRec = stoptime-starttime
    avg_tSignatureRec = tSignatureRec/len(critical_groups)

    
    starttime = time.time()
    if special!=1 and special!=2: # if we only need float16 or float32 standards then improve layer precision is not run
        #try: #Only need to improve precision if we need precision to be up to float64
        extracted_normal,extracted_bias,critical_groups, _ = check_quality(model,layerId-1, extracted_normal, extracted_bias,dimOfLayer,critical_groups, do_fix=True)
        extracted_normal, extracted_bias = improve_layer_precision(layerId-1,
                                                                    weights[:-1],biases[:-1],extracted_normal, extracted_bias,model,dimOfInput,dimOfPrevLayer,dimOfLayer,special,dataset,critical_groups)
        #except:
        #    pass
    stoptime = time.time()
    tImprovePrec = stoptime-starttime
    avg_tImprovePrec = tImprovePrec/len(critical_groups)

    query_count, crit_query_count = get_query_counts()
    avg_crit_query_count = int(crit_query_count/len(critical_groups))
    avg_query_count = int(query_count/len(critical_groups))
    print("Average time for critical point search and query count: ",avg_tFindCrt,tFindCrt,avg_crit_query_count,avg_query_count)
    print("Average time for signature recovery: ",avg_tSignatureRec,tSignatureRec)
    print("Average time for precision improvement: ",avg_tImprovePrec,tImprovePrec)

    return extracted_normal, extracted_bias, critical_groups,avg_tFindCrt,avg_tSignatureRec,avg_tImprovePrec,avg_query_count,avg_crit_query_count


# ----------
# Function for quality check. Called in neuronWiggle.py Signature Recovery part
# ----------
class DualStream:
    def __init__(self):
        self.console = sys.stdout
        self.stringio = StringIO()

    def write(self, message):
        self.console.write(message)
        self.stringio.write(message)

    def flush(self):
        self.console.flush()
        self.stringio.flush()

    def getvalue(self):
        return self.stringio.getvalue()
def check_quality(model, layer_num, extracted_normal, extracted_bias, dimOfLayer, critical_groups, do_fix=False):
    """
    Check the quality of the solution.
    
    The first function is read-only, and just reports how good or bad things are.
    The second half, when in cheating mode, will align the two matrices.
    """
# Redirect stdout to capture print statements while printing to console
    dual_stream = DualStream()
    old_stdout = sys.stdout
    sys.stdout = dual_stream

    try:
        print("\nCheck the solution of the last weight matrix.")
        
        reorder = [None]*(dimOfLayer)
        for i in range(dimOfLayer):
            gaps = []
            ratios = []
            for j in range(dimOfLayer):
                if np.all(np.abs(extracted_normal[:,i])) < 1e-9:
                    extracted_normal[:,i] += 1e-9
                ratio = model.get_layer(index=layer_num+1).get_weights()[0][:,j]/extracted_normal[:,i]
                ratio = np.median(ratio)
                error = model.get_layer(index=layer_num+1).get_weights()[0][:,j] - ratio * extracted_normal[:,i]
                error = np.sum(error**2)/np.sum(model.get_layer(index=layer_num+1).get_weights()[0][:,j]**2)
                gaps.append(error)
                ratios.append(ratio)
            print("Neuron", i, "maps on to neuron", np.argmin(gaps), "with error", np.min(gaps)**.5, 'ratio', ratios[np.argmin(gaps)])
            print("Bias check", (model.get_layer(index=layer_num+1).get_weights()[1][np.argmin(gaps)]-extracted_bias[i]*ratios[np.argmin(gaps)]))
            
            reorder[np.argmin(gaps)] = i
            if do_fix:
                extracted_normal[:,i] *= np.abs(ratios[np.argmin(gaps)])
                extracted_bias[i] *= np.abs(ratios[np.argmin(gaps)])
            
            if min(gaps) > 1e-2:
                print("ERROR LAYER EXTRACTED INCORRECTLY")
                print("\tGAPS:", " ".join("%.04f"%x for x in gaps))
                print("\t  Got:", " ".join("%.04f"%x for x in extracted_normal[:,i]/extracted_normal[0,i]))
                print("\t Real:", " ".join("%.04f"%x for x in model.get_layer(index=layer_num+1).get_weights()[0][:,np.argmin(gaps)]/model.get_layer(index=layer_num+1).get_weights()[0][0,np.argmin(gaps)]))


        # Randomly assign the unused neurons.
        used = [x for x in reorder if x is not None]
        missed = list(set(range(len(reorder))) - set(used))
        for i in range(len(reorder)):
            if reorder[i] is None:
                reorder[i] = missed.pop()  
        if do_fix and len(critical_groups)!=0:
            extracted_normal = extracted_normal[:,reorder]
            critical_groups = [critical_groups[i] for i in reorder]
            extracted_bias = extracted_bias[reorder]
        print("Real: ", model.get_layer(index=layer_num+1).get_weights())
        print("Corrected: ", extracted_normal, extracted_bias)
    finally:
        # Reset stdout
        sys.stdout = old_stdout

    # Get the string from DualStream object
    output = dual_stream.getvalue()
    return extracted_normal, extracted_bias, critical_groups, output

# ----------
# Helper functions for get_more_crit_pts which is the hyperplane following algorithm to find more critical points
# once we have found at least two critical points for each neuron
# ----------

def is_on_following_layer(model, A,B, known_A, known_B, point,dimInput,dimOfPrevLayer, dimOfLayer, special):
    if len(model.layers) == 3:
        GRAD_EPS = 1e1
    else:
        GRAD_EPS = 1e-4#1e-6#1e-4
    # Called in get_more_crit_pts
    print("Check if the critical point is on the next layer")
    
    def is_on_prior_layer(query):
        print("Hidden think", get_hidden_layers(query, A, B))
        if any(np.min(np.abs(layer)) < 1e-5 for layer in get_hidden_layers(query, A, B)):
            return True
        next_A, next_B = A+[known_A], B+[known_B]
        next_hidden = forward(query, next_A, next_B)
        #print(next_hidden)
        if np.min(np.abs(next_hidden)) < 1e-4:
            return True
        return False

    if is_on_prior_layer(point):
        print("It's not, because it's on an earlier layer")
        return False

    initial_signs = get_polytope_at(A, B, known_A, known_B, point)

    #normal = get_ratios(model,[point], A,B, eps=GRAD_EPS)[0].flatten()
    normal = get_ratios(model,[point], dimInput, eps=GRAD_EPS)[0].flatten()
    normal = normal / np.sum(normal**2)**.5

    for tol in range(10):
    
        random_dir = np.random.normal(size=dimInput)
        perp_component = np.dot(random_dir,normal)/(np.dot(normal, normal)) * normal
        parallel_dir = random_dir - perp_component
        
        go_direction = parallel_dir/np.sum(parallel_dir**2)**.5

        _, high = binary_search_towards(A, B,
                                        known_A, known_B,
                                        point,
                                        initial_signs,
                                        go_direction)

        point_in_same_polytope = point + (high * .999 - 1e-4) * go_direction

        print("high", high)

        solutions= do_better_sweep(model, point_in_same_polytope,
                            normal,
                            -1e-4 * high, 1e-4 * high)
        if len(solutions) >= 1:
            print("Correctly found", len(solutions))
        else:
            return False

        point_in_different_polytope = point + (high * 1.1 + 1e-1) * go_direction

        solutions = do_better_sweep(model, point_in_different_polytope,
                            normal,
                            -1e-4 * high, 1e-4 * high)
        if len(solutions) == 0:
            print("Correctly found", len(solutions))
        else:
            return False
        
    #print("I THINK IT'S ON THE NEXT LAYER")
        
    return True

PREV_GRAD = None
    
def binary_search_towards(A, B,  known_A, known_B, start_point, initial_signs, go_direction, maxstep=1e6):
    """
    Compute how far we can walk along the hyperplane until it is in a
    different polytope from a prior layer.

    It is okay if it's in a differnt polytope in a *later* layer, because
    it will still have the same angle.

    (but do it analytically by looking at the signs of the first layer)
    this requires no queries and could be done with math but instead
    of thinking I'm just going to run binary search.
    """
    global PREV_GRAD

    #_, slow_ans = binary_search_towards_slow(known_T,  known_A, known_B, start_point, initial_signs, go_direction, maxstep)
    new_A, new_B = A+[known_A],B+[known_B]
    # this is the hidden state
    initial_hidden = np.array(get_hidden_layers(start_point,new_A,new_B, flat=True))
    delta_hidden_np = (np.array(get_hidden_layers(start_point + 1e-6 * go_direction, new_A, new_B, flat=True)) - initial_hidden) * 1e6
    #
    #can_go_dist_all = initial_hidden / delta_hidden

    if PREV_GRAD is None or PREV_GRAD[0] is not A or PREV_GRAD[1] is not B or PREV_GRAD[2] is not known_A or PREV_GRAD[3] is not known_B:
        def get_grad(x, i):
            initial_hidden = get_hidden_layers(x, new_A, new_B, flat=True,np=jnp)
            return initial_hidden[i]
        g = jax.jit(jax.grad(get_grad))
        def grads(start_point, go_direction):
            return jnp.array([jnp.dot(g(start_point, i), go_direction) for i in range(initial_hidden.shape[0])])
        PREV_GRAD = (A, B, known_A, known_B, jax.jit(grads))
    else:
        grads = PREV_GRAD[4]
    
    delta_hidden = grads(start_point, go_direction)

    can_go_dist_all = np.array(initial_hidden / delta_hidden)
    
    can_go_dist = -can_go_dist_all[can_go_dist_all<0]

    if len(can_go_dist) == 0:
        print("Can't go anywhere at all")
        raise AcceptableFailure()

    can_go_dist = np.min(can_go_dist)
    
    a_bit_further = start_point + (can_go_dist+1e-4)*go_direction
    return a_bit_further, can_go_dist

def find_plane_angle(model,A, B,
                     known_A, known_B, dimInput,
                     multiple_intersection_point,
                     sign_at_init,
                     init_step,
                     exponential_base=1.5):
    """
    Given an input that's at the multiple intersection point, figure out how
    to continue along the path after it bends.


                /       X    : multiple intersection point
       ......../..      ---- : layer N hyperplane
       .      /  .       |   : layer N+1 hyperplane that bends
       .     /   .    
    --------X-----------
       .    |    .
       .    |    .
       .....|.....
            |
            |

    We need to make sure to bend, and not turn onto the layer N hyperplane.

    To do this we will draw a box around the X and intersect with the planes 
    and determine the four coordinates. Then draw another box twice as big.
    
    The first layer plane will be the two points at a consistent angle.
    The second layer plane will have an inconsistent angle.

    Choose the inconsistent angle plane, and make sure we move to a new
    polytope and don't just go backwards to where we've already been.
    """
    success = None
    camefrom = None

    prev_iter_intersections = []

    while True:
        x_dir_base = np.sign(np.random.normal(size=dimInput))/dimInput**.5
        y_dir_base = np.sign(np.random.normal(size=dimInput))/dimInput**.5
        # When the input dimension is odd we can't have two orthogonal
        # vectors from {-1,1}^DIM
        if np.abs(np.dot(x_dir_base, y_dir_base)) <= dimInput%2 + 1e-8:
            break

    MAX = 35

    start = [10] if init_step > 10 else []
    for stepsize in start + list(range(init_step, MAX)):
        print("\tTry stepping away", stepsize)
        x_dir = x_dir_base * (exponential_base**(stepsize-10))
        y_dir = y_dir_base * (exponential_base**(stepsize-10))
                
        # Draw the box as shown in the diagram above, and compute where
        # the critical points are.
        top = do_better_sweep(model,offset=multiple_intersection_point+x_dir,direction=y_dir, low=-1,high=1)
        bot = do_better_sweep(model,offset=multiple_intersection_point-x_dir,direction=y_dir,low=-1,high=1)
        left = do_better_sweep(model, offset=multiple_intersection_point + y_dir,
                               direction=x_dir, low=-1, high=1)
        right = do_better_sweep(model, offset=multiple_intersection_point - y_dir,
                                direction=x_dir, low=-1, high=1)
        intersections = top + bot + left + right

        # If we only have two critical points, and we're taking a big step,
        # then something is seriously messed up.
        # This is not an acceptable error. Just abort out and let's try to
        # do the whole thing again.
        if len(intersections) == 2 and stepsize >= 10:
            raise AcceptableFailure()

        if (len(intersections) == 0 and stepsize > 20): #if (len(intersections) == 0 and stepsize > 15):# or (len(intersections) == 3 and stepsize > 5):
            # Probably we're in just a constant flat 0 region
            # At this point we're basically dead in the water.
            # Just fail up and try again.
            print("\tIt looks like we're in a flat region, raise failure")
            raise AcceptableFailure()

        # If we somehow went from almost no critical points to more than 4,
        # then we've really messed up.
        # Just fail out and let's hope next time it doesn't happen.
        if len(intersections) > 4 and len(prev_iter_intersections) < 2:
            print("\tWe didn't get enough inner points")
            if exponential_base == 1.2:
                print("\tIt didn't work a second time")
                return None, None, 0
            else:
                print("\tTry with smaller step")
                return find_plane_angle(model,A, B,
                                        known_A, known_B, dimInput,
                                        multiple_intersection_point,
                                        sign_at_init,
                                        init_step, exponential_base=1.2)

        # This is the good, expected code-path.
        # We've seen four intersections at least twice before, and now
        # we're seeing more than 4.
        if (len(intersections) > 4 or stepsize > 20) and len(prev_iter_intersections) >= 2:
            next_intersections = np.array(prev_iter_intersections[-1])
            intersections = np.array(prev_iter_intersections[-2])

            # Let's first figure out what points are responsible for the prior-layer neurons
            # being zero, and which are from the current-layer neuron being zero
            candidate = []
            for i,a in enumerate(intersections):
                for j,b in enumerate(intersections):
                    if i == j: continue
                    score = np.sum(((a+b)/2-multiple_intersection_point)**2)
                    a_to_b = b-a
                    a_to_b /= np.sum(a_to_b**2)**.5

                    variance = np.std((next_intersections-a)/a_to_b,axis=1)
                    best_variance = np.min(variance)

                    candidate.append((best_variance, i, j))

            if sorted(candidate)[3][0] < 1e-8:
                # It looks like both lines are linear here
                # We can't distinguish what way is the next best way to go.
                print("\tFailed the box continuation finding procedure. (1)")
                #print("\t",candidate)
                raise AcceptableFailure()

            # Sometimes life is just ugly, and nothing wants to work.
            # Just abort.
            err, index_0, index_1 = min(candidate)
            if err/max(candidate)[0] > 1e-5:
                return None, None, 0

            prior_layer_near_zero = np.zeros(4, dtype=bool)
            prior_layer_near_zero[index_0] = True
            prior_layer_near_zero[index_1] = True

            # Now let's walk through each of these points and check that everything looks sane.
            should_fail = False
            for critical_point, is_prior_layer_zero in zip(intersections,prior_layer_near_zero):
                new_A, new_B = A+[known_A],B+[known_B]
                vs = get_hidden_layers(critical_point,new_A,new_B)

                if is_prior_layer_zero:
                    # We expect the prior layer to be zero.
                    if all([np.min(np.abs(x)) > 1e-5 for x in vs]):
                        # If it looks like it's not actually zero, then brutally fail.
                        print("\tAbort 1: failed to find a valid box")
                        should_fail = True
                if any([np.min(np.abs(x)) < 1e-10 for x in vs]):
                    # We expect the prior layer to be zero.
                    if not is_prior_layer_zero:
                        # If it looks like it's not actually zero, then brutally fail.
                        print("\tAbort 2: failed to find a valid box")
                        should_fail = True
            if should_fail:
                return None, None, 0
                
                

            # Done with error checking, life is good here.
            # Find the direction that corresponds to the next direction we can move in
            # and continue our search from that point.
            for critical_point, is_prior_layer_zero in zip(intersections,prior_layer_near_zero):
                sign_at_crit = sign_to_int(get_polytope_at(A, B,
                                                           known_A, known_B,
                                                           critical_point))
                print("\tMove to", sign_at_crit, 'versus', sign_at_init, is_prior_layer_zero)
                if not is_prior_layer_zero:
                    if sign_at_crit != sign_at_init:
                        success = critical_point
                    else:
                        camefrom = critical_point

            # If we didn't get a solution, then abort out.
            # Probably what happened here is that we got more than four points
            # on the box but didn't see exactly four points on the box twice before
            # this means we should decrease the initial step size and try again.
            if success is None:
                print("\tFailed the box continuation finding procedure. (2)")
                raise AcceptableFailure()
                #assert success is not None
            break
        if len(intersections) == 4:
            prev_iter_intersections.append(intersections)
    gc.collect()
    return success, camefrom, min(stepsize, MAX-3)

def sign_to_int(signs):
    """
    Convert a list to an integer.
    [-1, 1, 1, -1], -> 0b0110 -> 6
    """
    return int("".join('0' if x == -1 else '1' for x in signs),2)

def follow_hyperplane(LAYER, start_point, A,B, known_A, known_B, model, dimInput, dimOfPrevLayer, dimOfLayer, special,
                      history=[], MAX_POINTS=1e3, only_need_positive=False,target_neuron=None):
    """
    This is the ugly algorithm that will let us recover sign for expansive networks.
    Assumes we have extracted up to layer K-1 correctly, and layer K up to sign.

    start_point is a neuron on layer K+1

    known_T is the transformation that computes up to layer K-1, with
    known_A and known_B being the layer K matrix up to sign.

    We're going to come up with a bunch of different inputs,
    each of which has the same critical point held constant at zero.
    """

    def choose_new_direction_from_minimize(previous_axis):
        """
        Given the current point which is at a critical point of the next
        layer neuron, compute which direction we should travel to continue
        with finding more points on this hyperplane.

        Our goal is going to be to pick a direction that lets us explore
        a new part of the space we haven't seen before.
        """

        print("Choose a new direction to travel in")
        if len(history) == 0:
            which_to_change = 0
            new_perp_dir = perp_dir
            new_start_point = start_point
            initial_signs = get_polytope_at(A,B, known_A, known_B, start_point)

            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = min if initial_signs[0] == 1 else max
        else:
            neuron_values = np.array([x[1] for x in history])

            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)

            mean_plus_neuron_value = neuron_positive_count/(neuron_positive_count + neuron_negative_count + 1)
            mean_minus_neuron_value = neuron_negative_count/(neuron_positive_count + neuron_negative_count + 1)

            # we want to find values that are consistently 0 or 1
            # So map 0 -> 0 and 1 -> 0 and the middle to higher values
            if only_need_positive:
                neuron_consistency = mean_plus_neuron_value
            else:
                neuron_consistency = mean_plus_neuron_value * mean_minus_neuron_value

            # Print out how much progress we've made.
            # This estimate is probably worse than Windows 95's estimated time remaining.
            # At least it's monotonic. Be thankful for that.
            print("Progress", "%.1f"%int(np.mean(neuron_consistency!=0)*100)+"%")
            print("Counts on each side of each neuron")
            print(neuron_positive_count)
            print(neuron_negative_count)

            print("Neuron consistency: ",neuron_consistency)
            # Choose the smallest value, which is the most consistent
            which_to_change = np.argmin(neuron_consistency)
            
            print("Try to explore the other side of neuron", which_to_change)

            if which_to_change != previous_axis:
                if previous_axis is not None and neuron_consistency[previous_axis] == neuron_consistency[which_to_change]:
                    # If the previous thing we were working towards has the same value as this one
                    # the don't change our mind and just keep going at that one
                    # (almost always--sometimes we can get stuck, let us get unstuck)
                    which_to_change = previous_axis
                    new_start_point = start_point
                    new_perp_dir = perp_dir
                else:
                    valid_axes = np.where(neuron_consistency == neuron_consistency[which_to_change])[0]

                    best = (np.inf, None, None)

                    for _, potential_hidden_vector, potential_point in history[-1:]:
                        for potential_axis in valid_axes:
                            value = potential_hidden_vector[potential_axis]
                            if np.abs(value) < best[0]:
                                best = (np.abs(value), potential_axis, potential_point)

                    _, which_to_change, new_start_point = best
                    new_perp_dir = perp_dir
                    
            else:
                new_start_point = start_point
                new_perp_dir = perp_dir


            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = min if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else max
            arg_fn = np.argmin if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else np.argmax
            print("Changing", which_to_change, 'to flip sides because mean is', mean_plus_neuron_value[which_to_change])

        val = matmul(forward(new_start_point,A,B, with_relu=True), known_A, known_B)[which_to_change]

        initial_signs = get_polytope_at(A,B, known_A, known_B, new_start_point)

        # Now we're going to figure out what direction makes this biggest/smallest
        # this doesn't take any queries
        # There's probably an analytical way to do this.
        # But thinking is hard. Just try 1000 random angles.
        # There are no queries involved in this process.

        # Maybe one can make this more efficient??? -Hanna

        choices = []
        print("Figure out what direction makes this biggest/smallest")
        for _ in range(1000):
            random_dir = np.random.normal(size=dimInput)
            perp_component = np.dot(random_dir,new_perp_dir)/(np.dot(new_perp_dir, new_perp_dir)) * new_perp_dir
            parallel_dir = random_dir - perp_component

            # This is the direction we're going to travel in.
            go_direction = parallel_dir/np.sum(parallel_dir**2)**.5

            try:
                a_bit_further, high = binary_search_towards(A, B,
                                                            known_A, known_B,
                                                            new_start_point,
                                                            initial_signs,
                                                            go_direction)
            except AcceptableFailure:
                print("Binary search failed.")
                continue
            if a_bit_further is None:
                continue
            # choose a direction that makes the Kth value go down by the most
            val = matmul(forward(a_bit_further[np.newaxis,:], A, B, with_relu=True), known_A, known_B)[0][which_to_change]

            #print('\t', val, high)

            choices.append([val,
                            new_start_point + high*go_direction])

        if len(choices)==0:
            raise AcceptableFailure()
        best_value, multiple_intersection_point = fn(choices, key=lambda x: x[0])

        #print('Value', best_value)
        return new_start_point, multiple_intersection_point, which_to_change

    def check_relevant_crits(points_on_plane,neuron_values,neuron_positive_count):
        # Check if critical points are relevant and return only those that are relevant
        relevant_points_on_plane = []
        for i in range(len(points_on_plane)):
            relevant_neurons = neuron_positive_count <= 15
            points_contributing = (neuron_values[i] > 1) & relevant_neurons
            if np.any(points_contributing):
                relevant_points_on_plane.append(points_on_plane[i])
            else:
                mask = neuron_values[i] > 1
                neuron_positive_count[mask] -= 1

        return relevant_points_on_plane
    ###################################################
    ### Actual code to do the sign recovery starts. ###
    ###################################################
    start_box_step = 0
    points_on_plane = []
    current_change_axis = 0
    if target_neuron!=None:
        target_neuron_count = 0
        init_target_neuron_count = 0
    else:
        target_neuron_count = 1
        init_target_neuron_count = 0
    
    while True:
        gc.collect()
        print("\n\n")
        print("-----"*10)

        # Keep track of where we've been, so we can go to new places.
        which_polytope = get_polytope_at(A, B, known_A, known_B, start_point, False) # [-1 1 -1]
        hidden_vector = get_hidden_at(A, B, known_A, known_B, LAYER, start_point, False)
        sign_at_init = sign_to_int(which_polytope) # 0b010 -> 2

        print("Number of collected points", len(points_on_plane))
        if len(points_on_plane) > MAX_POINTS:
            if not GlobalConfig.set_Carlini:
                relevant_points_on_plane = check_relevant_crits(points_on_plane,neuron_values,neuron_positive_count)
                return relevant_points_on_plane, False
            else:
                return points_on_plane, False

        neuron_values = np.array([x[1] for x in history])

        neuron_positive_count = np.sum(neuron_values>1,axis=0)
        neuron_negative_count = np.sum(neuron_values<-1,axis=0)
        print("Positive count:", neuron_positive_count)
        print("Negative count:", neuron_negative_count)
        if target_neuron!=None:
            print("Target neuron: ",target_neuron)
            target_neuron_count = min(neuron_positive_count[target_neuron],neuron_negative_count[target_neuron])
            if len(points_on_plane)==0:
                init_target_neuron_count = target_neuron_count
            print("Target neuron count: ",target_neuron_count,init_target_neuron_count)

        if ((np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0)) or \
            (only_need_positive and np.all(neuron_positive_count > 0))) and \
            (target_neuron_count-init_target_neuron_count > 0):
            print("Have all the points we need (1)")
            print("Neuron positive count: ",neuron_positive_count)
            print("Neuron negative count: ",neuron_negative_count)

            neuron_values = np.array([get_hidden_at(A, B, known_A, known_B, LAYER, x, False) for x in points_on_plane])
            
            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)
            print("Neuron positive count: ",neuron_positive_count)
            print("Neuron negative count: ",neuron_negative_count)
            if not GlobalConfig.set_Carlini:
                relevant_points_on_plane = check_relevant_crits(points_on_plane,neuron_values,neuron_positive_count)
                return relevant_points_on_plane, True
            else:
                return points_on_plane, True
    
        # 1. find a way to move along the hyperplane by computing the normal
        # direction using the ratios function. Then find a parallel direction.
        
        try:
            #perp_dir = get_ratios([start_point], [range(DIM)], eps=1e-4)[0].flatten()
            perp_dir = get_ratios_lstsq(model, [start_point], A=[],B=[], eps=1e-5)[0].flatten()
        except AcceptableFailure:
            print("Failed to compute ratio at start point. Something very bad happened.")
            return points_on_plane, False

        # Record these points.
        history.append((which_polytope,
                        hidden_vector,
                        np.copy(start_point)))
        
        # We can't just pick any parallel direction. If we did, then we would
        # not end up covering much of the input space.
    
        # Instead, we're going to figure out which layer-1 hyperplanes are "visible"
        # from the current point. Then we're going to try and go reach all of them.
    
        # This is the point at which the first and second layers intersect.
        try:
            start_point, multiple_intersection_point, new_change_axis = choose_new_direction_from_minimize(current_change_axis)
        except AcceptableFailure:
            return points_on_plane, False
        if new_change_axis != current_change_axis:
            try:
                start_point, multiple_intersection_point, current_change_axis = choose_new_direction_from_minimize(None)
            except AcceptableFailure:
                return points_on_plane, False

        # Refine the direction we're going to travel in---stay numerically stable.
        towards_multiple_direction = multiple_intersection_point - start_point
        step_distance = np.sum(towards_multiple_direction**2)**.5

        print("Distance we need to step:", step_distance)        
        
        if step_distance > 1 or True:
            mid_point = 1e-4 * towards_multiple_direction/np.sum(towards_multiple_direction**2)**.5 + start_point
            
            mid_points = do_better_sweep(model, offset=mid_point, direction=perp_dir/np.sum(perp_dir**2)**.5, low=-1e-3, high=1e-3)
            if len(mid_points) > 0:
                mid_point = mid_points[np.argmin(np.sum((mid_point-mid_points)**2,axis=1))]

                towards_multiple_direction = mid_point - start_point
                towards_multiple_direction = towards_multiple_direction/np.sum(towards_multiple_direction**2)**.5

                initial_signs = get_polytope_at(A, B, known_A, known_B, start_point)
                try:
                    _, high = binary_search_towards(A, B,
                                                    known_A, known_B,
                                                    start_point,
                                                    initial_signs,
                                                    towards_multiple_direction)
                except AcceptableFailure:
                    print("Binary search failed.")
                    return points_on_plane, False

                multiple_intersection_point = towards_multiple_direction * high + start_point
                    
    
        # Find the angle of the next hyperplane
        # First, take random steps away from the intersection point
        # Then run the search algorithm to find some intersections
        # what we find will either be a layer-1 or layer-2 intersection.

        print("Now try to find the continuation direction")
        success = None
        while success is None:
            if start_box_step < 0:
                start_box_step = 0
                print("VERY BAD FAILURE")
                print("Choose a new random point to start from")
                which_point = np.random.randint(0, len(history))
                start_point = history[which_point][2]
                #print("New point is", which_point)
                current_change_axis = np.random.randint(0, dimOfLayer)
                print("New axis to change", current_change_axis)
                break

            print("\tStart the box step with size", start_box_step)
            try:
                success, camefrom, stepsize = find_plane_angle(model,A, B,
                                                               known_A, known_B, dimInput,
                                                               multiple_intersection_point,
                                                               sign_at_init,
                                                               start_box_step)
            except AcceptableFailure:
                # Go back to the top and try with a new start point
                print("\tOkay we need to try with a new start point")
                start_box_step = -10

            start_box_step -= 2

        if success is None:
            continue

        val = matmul(forward(multiple_intersection_point, A, B, with_relu=True), known_A, known_B)[new_change_axis]
        print("Value at multiple:", val)
        val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[new_change_axis]
        print("Value at success:", val)

        if stepsize < 10:
            new_move_direction = success - multiple_intersection_point
    
            # We don't want to be right next to the multiple intersection point.
            # So let's binary search to find how far away we can go while remaining in this polytope.
            # Then we'll go half as far as we can maximally go.
            
            initial_signs = get_polytope_at(A, B, known_A, known_B, success)
            print("polytope at initial", sign_to_int(initial_signs))        
            low = 0
            high = 1
            while high-low > 1e-2:
                mid = (high+low)/2
                query_point = multiple_intersection_point + mid * new_move_direction
                next_signs = get_polytope_at(A, B, known_A, known_B, query_point)
                print("polytope at", mid, sign_to_int(next_signs), "%x"%(sign_to_int(next_signs)^sign_to_int(initial_signs)))
                if initial_signs == next_signs:
                    low = mid
                else:
                    high = mid
            print("GO TO", mid)
        
            success = multiple_intersection_point + (mid/2) * new_move_direction
    
            val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[new_change_axis]
            print("Value at moved success:", val)
        
        print("Adding the points to the set of known good points")

        points_on_plane.append(start_point)
            
        if camefrom is not None:
            points_on_plane.append(camefrom)
        #print("Old start point", start_point)
        #print("Set to success", success)
        start_point = success
        start_box_step = max(stepsize-1,0)
        
    return points_on_plane, False

def get_more_crit_pts(A,B, known_A0, known_B0, critical_points, LAYER, model, dimInput,dimOfPrevLayer,dimOfLayer,special,
                     already_checked_critical_points=False,
                     only_need_positive=False, l1_mask=None,target_neuron=None):
    #Also cut the part that is not necessary for just generating new points
    # Do not need sign recovery here
    """
    Find more critical points for neurons in the target layer, when we have already found at least 2 critical point for each neuron.
    """
    def get_critical_points():
        print("Init")
        for point in critical_points:
            print("Tick")
            if already_checked_critical_points or is_on_following_layer(model, A,B, known_A0, known_B0, point,dimInput,dimOfPrevLayer,dimOfLayer, special):
                #print("Found layer N point at ", point, already_checked_critical_points)
                yield point

    get_critical_point = get_critical_points()


    print("Start looking for critical point")
    MAX_POINTS = 200
    which_point = next(get_critical_point)
    print("Done looking for critical point")

    initial_points = []
    history = []
    pts = []
    if already_checked_critical_points:
        for point in get_critical_point:
            initial_points.append(point)
            pts.append(point)
            which_polytope = get_polytope_at(A,B, known_A0, known_B0, point, False) # [-1 1 -1]
            hidden_vector = get_hidden_at(A,B, known_A0, known_B0, LAYER, point, False)
            history.append((which_polytope,
                            hidden_vector,
                            np.copy(point)))

    while True:
        if not already_checked_critical_points:
            history = []
            pts = []

        prev_count = -10
        good = False
        while len(pts) > prev_count+2:
            print("======"*10)
            print("RESTART SEARCH", len(pts), prev_count)
            #print(which_point)
            prev_count = len(pts)
            more_points, done = follow_hyperplane(LAYER, which_point,
                                        A,B,
                                        known_A0, known_B0, model,dimInput,dimOfPrevLayer,dimOfLayer,special,
                                        history=history,
                                        only_need_positive=only_need_positive,target_neuron=target_neuron)
            pts.extend(more_points)
            target_neuron = None
            if len(pts) >= MAX_POINTS:
                print("Have enough; break")
                break

            if len(pts) == 0:
                break
            new_A, new_B = A+[known_A0], B+[known_B0]
            neuron_values = forward(pts, new_A, new_B)                
            
            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)
            print("Counts")
            print(neuron_positive_count)
            print(neuron_negative_count)

            print("SHOULD BE DONE?", done, only_need_positive)
            if done and only_need_positive:
                good = True
                break
            if np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0) or \
               (only_need_positive and np.all(neuron_positive_count > 0)):
                print("Have all the points we need (2)")
                good = True
                break
            
        if len(pts) < MAX_POINTS/2 and good == False:
            print("======="*10)
            print("Select a new point to start from")
            print("======="*10)
            if already_checked_critical_points:
                print("CHOOSE FROM", len(initial_points))#, initial_points)
                if len(initial_points)>=2:
                    which_point = initial_points[np.random.randint(0,len(initial_points)-1)]
                else:
                    which_point = initial_points[0]
            else:
                which_point = next(get_critical_point)
        else:
            print("Abort")
            break
            
    critical_points = np.array(pts)#sorted(list(set(map(tuple,pts))))

    print("Now have critical points", len(critical_points))
        
    return critical_points


# ----------
# Helper functions for graph_solve and graph_solve
# ----------
@jax.jit
def process_block(ratios, other_ratios):
    # Let jax efficiently compute pairwise similarity by blocking things.
    
    differences = jnp.abs(ratios[:,jnp.newaxis,:] - other_ratios[jnp.newaxis,:,:])
    differences = differences / jnp.abs(ratios[:,jnp.newaxis,:]) + differences / jnp.abs(other_ratios[jnp.newaxis,:,:])

    close = differences < GlobalConfig.BLOCK_ERROR_TOL * jnp.log(ratios.shape[1])

    pairings = jnp.sum(close, axis=2) >= max(MIN_SAME_SIZE,BLOCK_MULTIPLY_FACTOR*(np.log(ratios.shape[1])-2))

    return pairings



def ratio_normalize(possible_matrix_rows):
    # We get a set of a bunch of numbers
    # a1 b1 c1 d1 e1 f1 g1 
    # a2 b2 c2 d2 e2 f2 g2
    # such that some of them are nan
    # We want to compute the pairwise ratios ignoring the nans
    ratio_evidence = [[[] for _ in range(possible_matrix_rows.shape[1])] for _ in range(possible_matrix_rows.shape[1])]

    for row in possible_matrix_rows:
        for i in range(len(row)):
            for j in range(len(row)):
                ratio_evidence[i][j].append(row[i]/row[j])

    if len(ratio_evidence) > 100:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float32)
    else:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float64) #float64
        
    medians = np.nanmedian(ratio_evidence, axis=2)
    errors = np.nanstd(ratio_evidence, axis=2) / np.sum(~np.isnan(ratio_evidence), axis=2)**.5
    errors += 1e-2 * (np.sum(~np.isnan(ratio_evidence), axis=2) == 1)
    errors /= np.abs(medians)
    errors[np.isnan(errors)] = 1e6

    ratio_evidence = medians

    # Choose the column with the fewest nans to return
    nancount = np.sum(np.isnan(ratio_evidence), axis=0)

    #print("Column nan count", nancount)
    
    column_ok = np.min(nancount) == nancount

    best = (None, np.inf)

    cost_i_over_j = ratio_evidence[:,:,np.newaxis]
    cost_j_over_k = ratio_evidence
    cost_i_over_k = cost_i_over_j * cost_j_over_k
    
    cost_i_j_k = cost_i_over_k
    # cost from i through j to k
    
    for column in range(len(column_ok)):
        if not column_ok[column]:
            continue

        quality = np.nansum(np.abs(cost_i_j_k[:,column,:] - ratio_evidence))
        #print('q',quality)
        if quality < best[1]:
            best = (column, quality)

    column, best_error = best
    
    return ratio_evidence[:,column], column, best_error

fixed_neurons = []

def graph_solve(all_ratios, all_criticals, expected_neurons, LAYER, debug=False):
    global fixed_neurons
    print("Length criticals group, all ratios group: ",len(all_ratios),len(all_criticals))
    print("Block multiply factor: ",BLOCK_MULTIPLY_FACTOR)
    # 1. Load the critical points and ratios we precomputed

    all_ratios = np.array(all_ratios, dtype=np.float64) #float64
    all_ratios_f32 = np.array(all_ratios, dtype=np.float32)#np.float32
    all_criticals = np.array(all_criticals, dtype=np.float64) #float64

    # Batch them to be sensibly sized
    ratios_group = [all_ratios_f32[i:i+1000] for i in range(0,len(all_ratios),1000)]
    criticals_group = [all_criticals[i:i+1000] for i in range(0,len(all_criticals),1000)]
                    
    # 2. Compute the similarity pairwise between the ratios we've computed

    print("Go up to", len(criticals_group))
    all_pairings = [[] for _ in range(sum(map(len,ratios_group)))]
    for batch_index,(criticals,ratios) in enumerate(zip(criticals_group, ratios_group)):
        print(batch_index)

        # Compute the all-pairs similarity
        axis = list(range(all_ratios.shape[1]))
        random.shuffle(axis)
        axis = axis[:20]
        for dim in axis:
            # We may have an error on one of the directions, so let's try all of them
            scaled_all_ratios =  all_ratios_f32 / all_ratios_f32[:,dim:dim+1]
            scaled_ratios = ratios / ratios[:,dim:dim+1]

            batch_pairings = process_block(scaled_ratios, scaled_all_ratios)
            
            # To get the offset, Compute the cumsum of the length up to batch_index
            batch_offset = sum(map(len,ratios_group[:batch_index]))
            # And now create the graph matching ratios that are similar
            for this_batch_i,global_j in zip(*np.nonzero(np.array(batch_pairings))):
                all_pairings[this_batch_i + batch_offset].append(global_j)
    #print("All pairings: ",all_pairings)
    graph = nx.Graph()
    # Add the edges to the graph, removing self-loops
    graph.add_edges_from([(i,j) for i,js in enumerate(all_pairings) for j in js if abs(i-j) > 1]) 
    components = list(nx.connected_components(graph))

    sorted_components = sorted(components, key=lambda x: -len(x))
    print("Sorted components: ",sorted_components)
            

    if len(components) == 0:
        print("No components found")
        raise AcceptableFailure(irr_idx=[],partial_solution=(np.array([]), np.array([])))
    print("Graph search found", len(components), "different components with the following counts", list(map(len,sorted_components)))

    previous_num_components = np.inf
    
    while previous_num_components > len(sorted_components):
        previous_num_components = len(sorted_components)
        candidate_rows = []
        candidate_components = []

        datas = [all_ratios[list(component)] for component in sorted_components]

        #This part is taking lots of time because of each pool init, would have to make it global but switch it to sequential for now
        #results =mp.Pool(MPROC_THREADS//4).map(ratio_normalize, datas)
        results = list(map(ratio_normalize, datas))

        candidate_rows = [x[0] for x in results]
        candidate_components = sorted_components

        candidate_rows = np.array(candidate_rows)

        new_pairings = [[] for _ in range(len(candidate_rows))]
        
        # Re-do the pairings
        for dim in range(all_ratios.shape[1]):
            scaled_ratios = candidate_rows / candidate_rows[:,dim:dim+1]

            batch_pairings = process_block(scaled_ratios, scaled_ratios)
            
            # And now create the graph matching ratios that are similar
            for this_batch_i,global_j in zip(*np.nonzero(np.array(batch_pairings))):
                new_pairings[this_batch_i].append(global_j)
        graph = nx.Graph()
        # Add the edges to the graph, ALLOWING self-loops this time
        graph.add_edges_from([(i,j) for i,js in enumerate(new_pairings) for j in js]) 
        components = list(nx.connected_components(graph))

        components = [sum([list(candidate_components[y]) for y in comp],[]) for comp in components]

        sorted_components = sorted(components, key=lambda x: -len(x))

        print("After re-doing the graph, the component counts is", len(components), "with items", list(map(len,sorted_components)))
    print("Processing each connected component in turn.")
    guessed_rows  = []

    resulting_examples = []
    resulting_rows = []
    
    skips_because_of_nan = 0
    failure = None

    irrelevant_indices = []

    for c_count, component in enumerate(sorted_components):
        possible_matrix_rows = all_ratios[list(component)]
        guessed_row, normalize_axis, normalize_error = ratio_normalize(possible_matrix_rows)
        guessed_rows.append([guessed_row, normalize_axis, normalize_error])
    if not GlobalConfig.set_Carlini:
        #def custom_sort(x): # Different way of sorting the clusters by number of nans instead of number of components in a cluster
        #    num_nans = np.isnan(x[1][0]).sum()
        #    component_length = len(x[0])
        #    # Adjust NaN count to give a slight advantage to components longer than 2
        #    adjusted_nan_count = num_nans - (component_length >= 3)
        #    return (adjusted_nan_count, -component_length)
        def custom_sort(x): #We sort by number of components in a cluster except for clusters where full signature is recovered. They are prioritized. This prevents and infinite loop that we previously had in Carlini's implementation.
            num_nans = np.isnan(x[1][0]).sum()
            component_length = len(x[0])
            if num_nans==0:
                component_length=np.inf
            return (-component_length, num_nans)
        paired_components = list(zip(sorted_components, guessed_rows))
        sorted_paired_components = sorted(paired_components, key=custom_sort)
        sorted_components = [comp for comp, _ in sorted_paired_components]
        guessed_rows = [guessed for _, guessed in sorted_paired_components]
    for c_count, component in enumerate(sorted_components):
        if debug:
            print("\n")
            if c_count >= expected_neurons:
                print("WARNING: This one might be a duplicate!")
        print("-----")
        print("On component", c_count, "with indexs", component)
        guessed_row, normalize_axis, normalize_error = guessed_rows[c_count]
        print('The guessed error in the computation is',normalize_error, 'with', len(component), 'witnesses')
        if normalize_error > .01 and len(component) <= 5:
            print("Component size less than 5 with high error; this isn't enough to be sure")
            continue
        print("Normalize on axis", normalize_axis)
        if len(resulting_rows):
            scaled_resulting_rows = np.array(resulting_rows)
            scaled_resulting_rows /= scaled_resulting_rows[:,normalize_axis:normalize_axis+1]
            delta = np.abs(scaled_resulting_rows - guessed_row[np.newaxis,:])
            # If I wanted to go back to check the scaling:
            #print("Scaled rows: ",scaled_resulting_rows)
            #print("Delta: ",delta)
            #print("Delta max, min: ",np.nanmax(delta, axis=1),min(np.nanmax(delta, axis=1)))
            if min(np.nanmax(delta, axis=1)) < 1e-2:
                print("Likely have found this node before")
                print("Should error be raised????")
                continue
        print("Guessed row: ",guessed_row)

        # If there is a nan value in a row, i.e., no full signature recovery for a neuron
        if np.any(np.isnan(guessed_row)) and c_count < expected_neurons: 
            print("Got NaN, need more data",len(component)/sum(map(len,components)),1/expected_neurons)
            if len(component) >= 3:
                # If less components than layer size (expected neurons) and more than 2 critical points for the concerned neuron
                if c_count < expected_neurons:
                    print("Component: ",c_count, " triggers GatherMoreData.")
                    nan_indices = np.argwhere(np.isnan(guessed_row)).flatten()
                    failure = (c_count, nan_indices[0],[all_criticals[x] for x in component])
                skips_because_of_nan += 1
            elif not GlobalConfig.set_Carlini:
                # If 2 or less critical points for the concerned neuron
                # Need to go into AcceptableFailure more random search
                failure = 'acceptable'
            continue
        elif not np.any(np.isnan(guessed_row)):
            # Memory deduplication
            if not GlobalConfig.set_Carlini:
                crits_fst_elements = [all_criticals[x][0] for x in component]
                found_set = False
                if fixed_neurons:
                    for i in range(len(fixed_neurons) - 1, -1, -1):
                        s = fixed_neurons[i]
                        # Calculate the intersection of component and s, and check its length
                        common_elements = set(crits_fst_elements).intersection(set(s[0]))
                        if len(common_elements) >= 5:
                            found_set = True
                            print("Normalize error difference: ",normalize_error,s[2])
                            all_crits_lst = list(all_criticals)
                            # We check in case the new result is better
                            if len(crits_fst_elements)<=len(s[0]) or (len(crits_fst_elements)<=len(s[0])+5 and normalize_error<s[2]*0.85) or (len(crits_fst_elements)<=len(s[0])+10 and normalize_error<s[2]*0.75) or (len(crits_fst_elements)<=len(s[0])+15 and normalize_error<s[2]*0.65) or (len(crits_fst_elements)<=len(s[0])+20 and normalize_error<s[2]*0.5):
                                pt_coords = list(set(s[0]) - set(crits_fst_elements)) # Will give me all elements in s[0] that are not in crits_fst_elements
                                value_to_index_map = {all_crits_lst[x][0]: x for x in s[1] if x < len(all_crits_lst) and all_crits_lst[x][0] in pt_coords}
                                # Here I want to remove s and instead append something else
                                del fixed_neurons[i]
                                fixed_neurons.append((set(crits_fst_elements),set(component),normalize_error))
                            else:
                                pt_coords = list(set(crits_fst_elements) - set(s[0])) # Will give me all elements in crits_fst_elements that are not in s[0]
                                value_to_index_map = {all_crits_lst[x][0]: x for x in component if x < len(all_crits_lst) and all_crits_lst[x][0] in pt_coords}
                            pt_coords_indices = [value_to_index_map[value] for value in pt_coords if value in value_to_index_map]
                            irrelevant_indices.extend(pt_coords_indices)
                            del all_crits_lst
                            break
                

                # If no similar set is found, add the new fixed neuron
                if not found_set and normalize_error<5e-3:
                    print("We are very sure about this neuron, so we save all critical points now and throw out all critical points that are found subsequently.")
                    print("Added new fixed neuron: ", crits_fst_elements)
                    fixed_neurons.append((set(crits_fst_elements),set(component),normalize_error))
            else:
                irrelevant_indices=[]       
        guessed_row[np.isnan(guessed_row)] = 0

        # Only for components which are bigger than 3 and which don't have np.isnan in guessed row and which are under expected_neurons count, we add them to resulting_rows => all neurons with full signature recovered
        if c_count < expected_neurons and len(component) >= 3:
            resulting_rows.append(guessed_row)
            resulting_examples.append([all_criticals[x] for x in component])
        else:
            print("Don't add it to the set")
    # We set failure when something went wrong but we want to defer crashing
    # (so that we can use the partial solution)
    gc.collect()
    print("All irrelevant indices: ",irrelevant_indices)
    print("len(all_ratios): ",len(all_ratios))
    print("len(resulting_rows): ",len(resulting_rows))
    
    # Here we either initiate more random search or we stop and output results
    if ((len(resulting_rows)+skips_because_of_nan < expected_neurons and len(all_ratios) < DEAD_NEURON_THRESHOLD) or failure=='acceptable'):
        print("We have not explored all neurons. Do more random search", len(resulting_rows), skips_because_of_nan, expected_neurons)
        raise AcceptableFailure(irr_idx=irrelevant_indices,partial_solution=(np.array(resulting_rows), resulting_examples))
    elif len(resulting_rows)!=0:
        print("At this point, we just assume the neuron must be dead")
        while len(resulting_rows) < expected_neurons: 
            resulting_rows.append(np.zeros_like((resulting_rows[0])))
            resulting_examples.append([np.zeros_like(resulting_examples[0][0])])

    # Here we know it's a GatherMoreData failure, but we want to only do this
    # if all neurons have been found up to DeadNeuronThreshold and we have at least 3 critical points for the target neuron
    # to initiate the faster targeted critical point search
    if failure is not None:
        print("Need to raise a previously generated failure.")
        print("Need more data for neuron ",failure[0])
        if GlobalConfig.set_Carlini:
            raise GatherMoreData((None,failure[2],irrelevant_indices))
        else:
            raise GatherMoreData((failure[1],failure[2],irrelevant_indices))
        
    print("Successfully returning a solution attempt.\n")
    return resulting_examples, resulting_rows

# ----------
# Helper functions for actual Signature Recovery
# ----------

def get_ratios(model, critical_points, dimInput, with_sign=True, eps=1e-5):
    """
    Compute the input weights to one neuron on the first layer.
    One of the core algorithms described in the paper.

    Given a set of critical point, compute the gradient for the first N directions.
    In practice N = range(DIM)

    Compute the second partial derivitive along each of the axes. This gives
    us the unsigned ratios corresponding to the ratio of the weights.

                      /
                  ^  /
                  | /
                  |/
             <----X---->  direction_1
                 /|
                / |
               /  V
              /  direction_2

    If we want to recover signs then we should also query on direction_1+direction_2
    And check to see if we get the correct solution.
    """
    weights, biases = getAllWeightsAndBiases(model)
    N=[range(dimInput)]
    ratios = []
    for j,point in enumerate(critical_points):
        ratio = []
        for i in N[j]:
            ratio.append(get_second_grad_unsigned(point, basis(i,dimInput),weights, biases, eps, eps/3))

        # Row sign recovery (a_1,a_2,...,a_n) -> recovery of sign of each a_i
        if with_sign:
            both_ratio = []
            for i in N[j]:
                both_ratio.append(get_second_grad_unsigned(point, (basis(i,dimInput) + basis(N[j][0],dimInput))/2,weights,biases, eps, eps/3))

            signed_ratio = []
            for i in range(len(ratio)):
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or 
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                positive_error = abs(abs(ratio[0]+ratio[i])/2 - abs(both_ratio[i]))
                negative_error = abs(abs(ratio[0]-ratio[i])/2 - abs(both_ratio[i]))

                if positive_error > 1e-4 and negative_error > 1e-4:
                    print("Probably something is borked")
                    print("d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))", positive_error, negative_error)
                    raise

                if positive_error < negative_error:
                    signed_ratio.append(ratio[i])
                else:
                    signed_ratio.append(-ratio[i])
        else:
            signed_ratio = ratio
        
        ratio = np.array(signed_ratio)

        #print(ratio)
        ratios.append(ratio)
        
    return ratios

def get_ratios_lstsq(model, critical_points, A, B, debug=False, eps=1e-5):
    """
    Do the same thing as get_ratios, but works when we can't directly control where we want to query.
    
    This means we can't directly choose orthogonal directions, and so we're going
    to just pick random ones and then use least-squares to do it
    """
    weights,biases = getAllWeightsAndBiases(model)
    ratios = []
    for i,point in enumerate(critical_points):
        # We're going to create a system of linear equations
        # d_matrix is going to hold the inputs,
        # and ys is going to hold the resulting learned outputs
        d_matrix = []
        ys = []

        # Query on N+2 random points, so that we have redundency
        # for the least squares solution.
        for i in range(np.sum(forward(point,A,B) != 0)+2):
            # 1. Choose a random direction
            d = np.sign(np.random.normal(0,1,point.shape))
            d_matrix.append(d)

            # 2. See what the second partial derivitive at this value is
            ratio_val = get_second_grad_unsigned(point, d,weights,biases, eps, eps/3)

            # 3. Get the sign correct
            if len(ys) > 0:
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or 
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                both_ratio_val = get_second_grad_unsigned(point, (d+d_matrix[0])/2,weights,biases, eps, eps/3)

                positive_error = abs(abs(ys[0]+ratio_val)/2 - abs(both_ratio_val))
                negative_error = abs(abs(ys[0]-ratio_val)/2 - abs(both_ratio_val))

                if positive_error > 1e-4 and negative_error > 1e-4:
                    print("Probably something is borked")
                    print("d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))", positive_error, negative_error)
                    raise AcceptableFailure()

                
                if negative_error < positive_error:
                    ratio_val *= -1
            
            ys.append(ratio_val)

        d_matrix = np.array(d_matrix)
        # Now we need to compute the system of equations
        # We have to figure out what the vectors look like in hidden space,
        # so compute that precisely
        h_matrix = np.array(forward_at(point,A,B, d_matrix))

            
        # Which dimensions do we lose?
        column_is_zero = np.mean(np.abs(h_matrix)<1e-8,axis=0) > .5
        assert np.all((forward(point,A,B, with_relu=True) == 0) == column_is_zero)

        # Solve the least squares problem and get the solution
        # This is equal to solving for the ratios of the weight vector
        soln, *rest = np.linalg.lstsq(np.array(h_matrix, dtype=np.float64), #float64 #float32
                                      np.array(ys, dtype=np.float64), 1e-5) #float64 #float32
    
        # Set the columns we know to be wrong to NaN so that it's obvious
        # this isn't important but it helps us distinguish from genuine errors
        # and the kind that we can't avoic because of zero gradients
        soln[column_is_zero] = np.nan

        ratios.append(soln)
        
    return ratios
def gather_ratios(critical_points, A ,B ,check_fn, LAYER, COUNT, model,dimInput,dimOfPrevLayer,dimOfLayer, special, eps=1e-6):
    """
    Calculate the ratio values/partial signatures that can be deduced from each critical point.
    """
    this_layer_critical_points = []
    print("Gathering", COUNT, "critical points")
    for point in critical_points:
        if LAYER > 0:
            #Check if the output of any hidden layer before relu is less than tolerance, if yes we move on
            #i.e., the output is very small, this means further computations might fail due to limitation in precision?
            if any(np.any(np.abs(x) < 1e-5) for x in get_hidden_layers(point,A,B)): #Maybe adjust to float64 to 1e-12?
                continue
            
        if LAYER > 0 and np.sum(forward(point,A,B) != 0 ) <= 1:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            #print("Check function rejected it")
            continue
        if len(model.layers) == 3:
            GRAD_EPS = 1e1
        else:
            GRAD_EPS = 1e-4#1e-6#1e-4
        for EPS in [GRAD_EPS, GRAD_EPS/10, GRAD_EPS/100]:
            try:
                normal = get_ratios_lstsq(model, [point], A,B, eps=EPS)[0].flatten()
                #normal = get_ratios([point], [range(DIM)], eps=EPS)[0].flatten()
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                continue
        try:
            this_layer_critical_points.append((normal, point))
        except:
            continue
        # coupon collector: we need nlogn points.
        print("Up to", len(this_layer_critical_points), 'of', COUNT)
        if len(this_layer_critical_points) >= COUNT:
            break
    gc.collect()
    return this_layer_critical_points
def compute_layer_values(critical_points,model, weights,biases,layerId,dimInput,dimOfPrevLayer,dimOfLayer,special):
    """
    Main function of Signature Recovery that puts everything together.
    It outputs the full signature for all neurons in a layer in the end if it completes succesfully.
    """
    tCritExtra = 0.0
    LAYER = layerId-1
    if LAYER == 0:
        COUNT = dimOfLayer * 3
    else:
        COUNT = dimOfLayer * np.log(dimOfLayer) * 3


    # type: [(ratios, critical_point)]
    this_layer_critical_points = []

    partial_weights = None
    partial_biases = None
    A,B = weights[:-1],biases[:-1]

    def check_fn(point):
        if partial_weights is None:
            return True
        hidden = matmul(forward(point,A,B, with_relu=True), partial_weights.T, partial_biases)
        if np.any(np.abs(hidden) < 1e-4):
            return False
        return True

    print()
    print("Start running critical point search to find neurons on layer", LAYER)

    while True:
        starttime = time.time()
        print("At this iteration I have", len(this_layer_critical_points), "critical points")
        #Do not really see much point to have this function?? Same as just iterating over critical_points id say
        def reuse_critical_points():
            for witness in critical_points:
                yield witness
        this_layer_critical_points.extend(gather_ratios(reuse_critical_points(), A ,B ,check_fn,
                      LAYER, COUNT, model,dimInput,dimOfPrevLayer,dimOfLayer, special)) ##1

        print("And now up to ", len(this_layer_critical_points), "critical points")
        stoptime = time.time()
        tCritExtra += stoptime-starttime

        ## filter out duplicates
        filtered_points = []

        # Let's not add points that are identical to ones we've already done.
        # Seems like it will take lots of time --> maybe instead could do this later for partial signature computation, whether or not to do it

        for i,(ratio1,point1) in enumerate(this_layer_critical_points):
            #Probably if this does not work we did not get a ratio for a point
            #print("This layer critical points[i+1:]: ",this_layer_critical_points.shape)
            for args in this_layer_critical_points[i+1:]:
                try: 
                    ratio2,point2 = args
                except:
                    print("Some error in line compute_layer_values")
                    continue
                if np.sum((point1 - point2)**2)**.5 < 1e-10:
                    break
            else:
                filtered_points.append((ratio1, point1))

        
        this_layer_critical_points = filtered_points

        print("After filtering duplicates we're down to ", len(this_layer_critical_points), "critical points")
        

        print("Start trying to do the graph solving")
        try:
            critical_groups, extracted_normals = graph_solve([x[0] for x in this_layer_critical_points],
                                                             [x[1] for x in this_layer_critical_points],
                                                             dimOfLayer,
                                                             LAYER=LAYER,
                                                             debug=True)
            break
        except GatherMoreData as e:
            print("Graph solving failed because we didn't explore all sides of at least one neuron")
            print("Fall back to the hyperplane following algorithm in order to get more data")
            irrelevant_indices_set = set(e.data[2])
            this_layer_critical_points = [point for idx, point in enumerate(this_layer_critical_points) if idx not in irrelevant_indices_set]

            starttime = time.time()
            def mine(r):
                while len(r) > 0:
                    print("Yielding a point")
                    yield r[0]
                    r = r[1:]
                print("No more to give!")

            print("LAYER: ",LAYER)
            prev_A,prev_B = A[:-1],B[:-1]
            print("e.data[0]: ",e.data[0])
            more_critical_points = get_more_crit_pts(prev_A, prev_B, A[-1], B[-1], mine(e.data[1]),
                                                                     LAYER-1, model, dimInput, dimOfPrevLayer, dimOfLayer, special, already_checked_critical_points=True,
                                                                     only_need_positive=True, target_neuron=e.data[0])
            stoptime = time.time()
            tCritExtra += stoptime-starttime
            print("Add more", len(more_critical_points))
            more_ratios = gather_ratios(more_critical_points,A,B,check_fn,LAYER, 1e6, model,dimInput,dimOfPrevLayer,dimOfLayer, special)
            this_layer_critical_points.extend(more_ratios)##2
            print("Done adding")
            
            COUNT = dimOfLayer
        except AcceptableFailure as e:
            irrelevant_indices_set = set(e.irr_idx)
            this_layer_critical_points = [point for idx, point in enumerate(this_layer_critical_points) if idx not in irrelevant_indices_set]

            print("Graph solving failed; get more points")
            COUNT = dimOfLayer
            if 'partial_solution' in dir(e):

                if len(e.partial_solution[0]) > 0:
                    partial_weights, corresponding_examples = e.partial_solution
                    print("Got partial solution with shape", partial_weights.shape)
                        
                    partial_biases = []
                    for weight, examples in zip(partial_weights, corresponding_examples):
                        examples = np.array(examples)
                        hidden = forward(examples,A,B, with_relu=True)
                        print("hidden", np.array(hidden).shape)
                        bias = -np.median(np.dot(hidden, weight))
                        partial_biases.append(bias)
                    partial_biases = np.array(partial_biases)

    print("Number of critical points per cluster", [len(x) for x in critical_groups])
    
    point_per_class = [x[0] for x in critical_groups]

    extracted_normals = np.array(extracted_normals).T
    print("Extracted normal: ",extracted_normals)
    print("Extracted normals shape: ",extracted_normals.shape)

    # Compute the bias because we know wx+b=0
    extracted_bias = [matmul(forward(point_per_class[i],A,B, with_relu=True), extracted_normals[:,i], c=None) for i in range(dimOfLayer)]

    # Don't forget to negate it.
    # That's important.
    # No, I definitely didn't forget this line the first time around.
    extracted_bias = -np.array(extracted_bias)

    # For the failed-to-identify neurons, set the bias to zero
    extracted_bias *= np.any(extracted_normals != 0,axis=0)[:,np.newaxis]
    print("Extracted bias: ",extracted_bias)

    return extracted_normals, extracted_bias, critical_groups,tCritExtra
