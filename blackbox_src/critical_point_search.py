import numpy as np
import random
from .global_vars import GlobalConfig
from .utils import getAllWeightsAndBiases, predict_manual_fast
# ==========
#  Functions for Basic Critical Point Search
# ==========

def getCIFARtestImage(batch_size=1): 
    """Pick a random flattened input image from CIFAR."""
    
    import tensorflow as tf
    from keras.datasets import cifar10

    def normalize_resize(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255)
        image = tf.image.resize(image, (32,32))
        return image, label

    (trainX, trainy), (testX, testy) = cifar10.load_data()

    #trainX, trainy = normalize_resize(trainX, trainy)
    testX, testy = normalize_resize(testX, testy)
    # Flatten the test dataset
    testX = tf.keras.layers.Flatten()(testX)
    # pick a random input image
    idx = np.random.choice(np.arange(len(testX)), size=batch_size, replace=False)
    idx_tensor = tf.convert_to_tensor(idx, dtype=tf.int32)
    # Use the tensor for indexing
    batch_images = tf.gather(testX, indices=idx_tensor)
    batch_labels = tf.gather(testy, indices=idx_tensor)
    if batch_size==1:
        return batch_images.numpy().flatten()
    else:
        return batch_images.numpy()
def getMNISTtestImage(batch_size=1, special_setting=False): 
    """Pick a random flattened input image from CIFAR."""
    
    import tensorflow as tf
    from keras.datasets import mnist

    def normalize_resize(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255)
        return image, label

    (trainX, trainy), (testX, testy) = mnist.load_data()

    if special_setting==True:
        #Only take two of each class to test whether having two of each class will help the critical point search
        selected_images = []
        selected_labels = []
        images_per_label = 2
        label_count = {label: 0 for label in range(10)}
        for image, label in zip(testX, testy):
            if label_count[label] < images_per_label:
                selected_images.append(image)
                selected_labels.append(label)
                label_count[label] += 1
            if all(count == images_per_label for count in label_count.values()):
                break
        testX = np.array(selected_images)
        testy = np.array(selected_labels)

    testX, testy = normalize_resize(testX, testy)
    # Flatten the test dataset
    testX = tf.keras.layers.Flatten()(testX)
    # pick a random input image
    idx = np.random.choice(np.arange(len(testX)), size=batch_size, replace=False)
    idx_tensor = tf.convert_to_tensor(idx, dtype=tf.int32)
    # Use the tensor for indexing
    batch_images = tf.gather(testX, indices=idx_tensor)
    batch_labels = tf.gather(testy, indices=idx_tensor)
    if batch_size==1:
        return batch_images.numpy().flatten()
    else:
        return batch_images.numpy()

def sweep_for_critical_points(model, dimInput, std=1,data=None,setting='original'):
    print("Sweep for critical points")
    while True:
        print("Start another sweep")
        if data!='mnist':
            sweep = do_better_sweep(model,offset=np.random.normal(0, np.random.uniform(std/10,std), size=dimInput), low=-std*1e3, high=std*1e3)
        elif data=='mnist' and setting=='WithDataPoints0.5': # In case we want to test if performance is better if we have a few data points from mnist dataset
            if random.random() < 0.5 or GlobalConfig.query_count>100000:
                sweep = do_better_sweep(model, low=-std*1e3, high=std*1e3,dataset=data)
            else:
                sweep = do_better_sweep(model, offset=getMNISTtestImage(1,special_setting=True),direction=np.random.normal(0,1,size=dimInput).flatten()*(8/255), low=-1, high=1,dataset=data)
        else: #data=='mnist'
            sweep = do_better_sweep(model, low=-std*1e3, high=std*1e3,dataset=data)
        print("Total intersections found", len(sweep))
        for point in sweep:
            yield point

def do_better_sweep(model, offset=None, direction=None, low=-1e3, high=1e3, dataset=None):
    """
    A much more efficient implementation of searching for critical points.
    Has the same interface as do_slow_sweep.

    Nearly identical, except that when we are in a region with only one critical
    point, does some extra math to identify where exactly the critical point is
    and returns it all in one go.
    In practice this is both much more efficient and much more accurate.
    
    """
    if len(model.layers) == 3:
        SKIP_LINEAR_TOL = 1e-7
    else:
        SKIP_LINEAR_TOL = 1e-8
    shape = model.input_shape[1:]

    if offset is None:
        offset = np.random.normal(0,1,size=shape).flatten()
    if direction is None:
        direction = np.random.normal(0,1,size=shape).flatten()
    weights,biases = getAllWeightsAndBiases(model)
    def memo_forward_pass(x, c={}):
        if x not in c:
            c[x] = predict_manual_fast((offset+direction*x)[np.newaxis, :],weights,biases)
        return c[x]
    relus = []
    def search(low, high):
        GlobalConfig.crit_query_count+=1
        mid = (low+high)/2
        y1 = f_low = memo_forward_pass(low)
        f_mid = memo_forward_pass(mid) 
        y2 = f_high = memo_forward_pass(high)
        if np.abs(f_mid - (f_high + f_low)/2) < SKIP_LINEAR_TOL*((high-low)**.5):
            return
        elif high-low < 1e-8:
            # Too close to each other
            return
        else:
            # Check if there is exactly one ReLU switching sign, or if there are multiple.
            # To do this, use the 2-linear test from Jagielski et al. 2019
            #
            #            
            #             /\   <---- real_h_at_x
            #            /  \
            #           /    \
            #          /      \
            #         /        \
            #        /          \
            #       /            \
            #     low q1 x_s_b q3 high
            # 
            # Use (low,q1) to estimate the direction of the first line
            # Use (high,q3) to estimate the direction of the second line
            # They should in theory intersect at (x_should_be, y_should_be)
            # Query to compute real_h_at_x and then check if that's what we get
            # Then check that we're linear from x_should_be to low, and
            # linear from x_should_be to high.
            # If it all checks out, then return the solution.
            # Otherwise recurse again.
            q1 = (low+mid)*.5
            q3 = (high+mid)*.5

            f_q1 = memo_forward_pass(q1)
            f_q3 = memo_forward_pass(q3)

            
            m1 = (f_q1-f_low)/(q1-low)
            m2 = (f_q3-f_high)/(q3-high)

            if m1 != m2:
                d = (high-low)
                alpha = (y2 - y1 - d * m2) / (d * m1 - d * m2)
                
                x_should_be = low + (y2 - y1 - d * m2) / (m1 - m2)
                height_should_be = y1 + m1*(y2 - y1 - d * m2) / (m1 - m2)
            
            if m1 == m2:
                # If the slopes on both directions are the same (e.g., the function is flat)
                # then we need to split and can't learn anything
                pass
            elif np.all(.25+1e-5 < alpha) and np.all(alpha < .75-1e-5) and np.max(x_should_be)-np.min(x_should_be) < 1e-5:
                x_should_be = np.median(x_should_be)
                real_h_at_x = memo_forward_pass(x_should_be)
                
                if np.all(np.abs(real_h_at_x - height_should_be) < SKIP_LINEAR_TOL*100):
                    # Compute gradient on each side and check for linearity 
                    eighth_left =  x_should_be-1e-4
                    eighth_right = x_should_be+1e-4
                    grad_left = (memo_forward_pass(eighth_left)-real_h_at_x)/(eighth_left-x_should_be)
                    grad_right = (memo_forward_pass(eighth_right)-real_h_at_x)/(eighth_right-x_should_be)

                    if np.all(np.abs(grad_left-m1)>SKIP_LINEAR_TOL*10) or np.all(np.abs(grad_right-m2)>SKIP_LINEAR_TOL*10):
                        #print("it's nonlinear")
                        pass
                    else:
                        relus.append(offset + direction*x_should_be)
                        return
            
        search(low, mid)
        search(mid, high)

    search(low,
           high)

    return relus


