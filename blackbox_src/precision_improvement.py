import gc
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .utils import forward, matmul, get_hidden_layers, AcceptableFailure
from .critical_point_search import do_better_sweep
# ==========
#  Functions for Precision Improvement
# ==========
def improve_row_precision_Carlini(args):
    """
    Improve the precision of an extracted row.
    We think we know where it is, but let's actually figure it out for sure.

    To do this, start by sampling a bunch of points near where we expect the line to be.
    This gives us a picture like this

                      X
                       X
                    
                   X
               X
                 X
                X

    Where some are correct and some are wrong.
    With some robust statistics, try to fit a line that fits through most of the points
    (in high dimension!)

                      X
                     / X
                    / 
                   X
               X  /
                 /
                X

    This solves the equation and improves the point for us.
    """
    gc.collect()
    (LAYER, A,B, known_A, known_B, row, model, dimInput, dimOfPrevLayer, dimOfLayer, special, dataset, did_again,prior_found_crit_pts) = args
    print("Improve the extracted neuron number", row)

    if np.sum(np.abs(known_A[:,row])) < 1e-8:
        return known_A[:,row], known_B[row]
        

    def loss(x, r):
        hidden = forward(x,A,B, with_relu=True, np=jnp)
        dotted = matmul(hidden, jnp.array(known_A)[:,r], jnp.array(known_B)[r], np=jnp)
        return jnp.sum(jnp.square(dotted))

    loss_grad = jax.jit(jax.grad(loss))
    loss = jax.jit(loss)

    extended_A,extended_B = A+[known_A],B+[known_B]

    def get_more_points(NUM,found_crit_pts):
        """
        Gather more points. This procedure is really kind of ugly and should probably be fixed.
        We want to find points that are near where we expect them to be.

        So begin by finding preimages to points that are on the line with gradient descent.
        This should be completely possible, because we have d_0 input dimensions but 
        only want to control one inner layer.
        """
        print("Gather some more actual critical points on the plane")
        @jax.jit
        def update(step, opt_state, points, row):
            grads = loss_grad(points, row)
            updates, opt_state = optimizer.update(grads, opt_state)
            points = optax.apply_updates(points, updates)
            return points, opt_state
        stepsize = .1
        critical_points = []
        while len(critical_points) <= NUM:
            print("On this iteration I have ", len(critical_points), "critical points on the plane")
            points = np.random.normal(0, 1e3, size=(100,dimInput,))
            lr = 10
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(points)
            for step in range(5000): #5000
                # Use JaX's built in optimizer to do this.
                # We want to adjust the LR so that we get a better solution
                # as we optimize. Probably there is a better way to do this,
                # but this seems to work just fine.

                # No queries involvd here.
                if step%1000 == 0:
                    lr *= .5
                    optimizer = optax.adam(lr)
                    opt_state = optimizer.init(points)
                if step%100 == 0:
                    ell = loss(points, row)
                    if ell < 1e-5:
                        break
                points, opt_state = update(step, opt_state, points, row)
            for point in points:
                # For each point, try to see where it actually is.

                # First, if optimization failed, then abort.
                if loss(point, row) > 1e-5: 
                    continue

                if LAYER > 0:
                    # If we're on a deeper layer, and if a prior layer is zero, then abort
                    if min(np.min(np.abs(x)) for x in get_hidden_layers(point,A,B)) < 1e-4:
                        print("is on prior")
                        continue
                    
                solution = do_better_sweep(model, offset=point,low=-stepsize, high=stepsize, dataset = dataset)

                if len(solution) == 0:
                    stepsize *= 1.1
                elif len(solution) > 1:
                    stepsize /= 2
                elif len(solution) == 1:
                    stepsize *= 0.98
                    potential_solution = solution[0]

                    hiddens = get_hidden_layers(potential_solution,extended_A,extended_B)
                    this_hidden_vec = forward(potential_solution,extended_A,extended_B)
                    this_hidden = np.min(np.abs(this_hidden_vec))
                    if min(np.min(np.abs(x)) for x in this_hidden_vec) > np.abs(this_hidden)*0.9:
                        critical_points.append(potential_solution)
                    else:
                        print("Reject it")
        print("Finished with a total of", len(critical_points), "critical points")
        return critical_points


    critical_points_list = []
    for _ in range(1):
        NUM = dimOfPrevLayer*2
        try:
            critical_points_list.extend(get_more_points(NUM,prior_found_crit_pts))
        except AcceptableFailure:
            continue

        critical_points = np.array(critical_points_list)

        hidden_layer = forward(np.array(critical_points),A,B, with_relu=True)

        crit_val_1 = matmul(hidden_layer, known_A[:,row], known_B[row])

        best = (None, 1e6)
        upto = 100#100

        for iteration in range(upto):
            if iteration%1000 == 0:
                print("ITERATION", iteration, "OF", upto)
            if iteration%2 == 0 or True:

                # Try 1000 times to make sure that we get at least one non-zero per axis
                for _ in range(1000):
                    if NUM + 2 > len(hidden_layer):
                        print(f"Warning: Requested sample size (NUM+2) exceeds the size of hidden_layer. Adjusting NUM: ", NUM+2, len(hidden_layer))
                        NUM = max(0, len(hidden_layer) - 2)  # Adjust NUM to fit the available size
                    randn = np.random.choice(len(hidden_layer), NUM+2, replace=False)
                    if np.all(np.any(hidden_layer[randn] != 0, axis=0)):
                        break

                hidden = hidden_layer[randn]
                soln,*rest = np.linalg.lstsq(hidden, np.ones(hidden.shape[0]))
            
            crit_val_2 = matmul(hidden_layer, soln, None)-1
            
            quality = np.median(np.abs(crit_val_2))

            if iteration%100 == 0:
                print('quality', quality, best[1])
            if quality < best[1] and not np.any(np.abs(soln) < 1e-5): # Checks that solution is viable
                best = (soln, quality)
                #print("Crit val 2: ",crit_val_2)

            if quality < 1e-10: break
            if quality < 1e-10 and iteration > 1e4: break
            if quality < 1e-8 and iteration > 1e5: break

        soln, _ = best

        print("Compare",
              np.median(np.abs(crit_val_1)),
              best[1])
        #print("Crit val 1: ",crit_val_1)
        if best[0] is None:
            soln = known_A[:, row]

        if np.all(np.abs(soln) > 1e-10):
            break

    print('soln',soln)
    
    if np.any(np.abs(soln) < 1e-10):
        print("THIS IS BAD. FIX ME NOW.")
        raise
    
    rescale = np.median(soln/known_A[:,row])
    soln[np.abs(soln) < 1e-10] = known_A[:,row][np.abs(soln) < 1e-10] * rescale


    
    if best[1] < np.mean(np.abs(crit_val_1)) or True:
        return soln, -1 ### Recalculate bias?
    else:
        print("FAILED TO IMPROVE ACCURACY OF ROW", row)
        print(np.mean(np.abs(crit_val_2)), 'vs', np.mean(np.abs(crit_val_1)))
        return known_A[:,row], known_B[row]

def improve_row_precision_ours(args):
    # This is supposed to work for mnist models as well but sorry its not great.
    # Since it is not necessarily needed not that much effort was spent on optimising this.
    # It can definitely be optimised in the critical point search process ㅠㅠ
    """
    Improve the precision of an extracted row.
    We think we know where it is, but let's actually figure it out for sure.

    To do this, start by sampling a bunch of points near where we expect the line to be.
    This gives us a picture like this

                      X
                       X
                    
                   X
               X
                 X
                X

    Where some are correct and some are wrong.
    With some robust statistics, try to fit a line that fits through most of the points
    (in high dimension!)

                      X
                     / X
                    / 
                   X
               X  /
                 /
                X

    This solves the equation and improves the point for us.
    """
    gc.collect()
    (LAYER, A,B, known_A, known_B, row, model, dimInput, dimOfPrevLayer, dimOfLayer, special, dataset, did_again,prior_found_crit_pts) = args
    print("Improve the extracted neuron number", row)

    print(np.sum(np.abs(known_A[:,row])))
    if np.sum(np.abs(known_A[:,row])) < 1e-8:
        return known_A[:,row], known_B[row]
        

    def loss(x, r):
        hidden = forward(x,A,B, with_relu=True, np=jnp)
        dotted = matmul(hidden, jnp.array(known_A)[:,r], jnp.array(known_B)[r], np=jnp)
        return jnp.sum(jnp.square(dotted))+jnp.max(jnp.square(dotted))

    loss_grad = jax.jit(jax.grad(loss))
    loss = jax.jit(loss)

    def get_more_points(NUM,found_crit_pts):
        """
        Gather more points. This procedure is really kind of ugly and should probably be fixed.
        We want to find points that are near where we expect them to be.

        So begin by finding preimages to points that are on the line with gradient descent.
        This should be completely possible, because we have d_0 input dimensions but 
        only want to control one inner layer.
        """
        print("Gather some more actual critical points on the plane")
        @jax.jit
        def update(step, opt_state, points, row):
            grads = loss_grad(points, row)
            updates, opt_state = optimizer.update(grads, opt_state)
            points = optax.apply_updates(points, updates)
            return points, opt_state
        stepsize = .1
        critical_points = []
        count = 0
        while len(critical_points) <= NUM+10:#NUM+1:
            count+=1
            if count>=100:
                print("Failure after 100 iterations.")
                raise AcceptableFailure()
            print("On this iteration I have ", len(critical_points), "critical points on the plane of ", NUM+11)
            
            points = np.array(found_crit_pts) # Reuse the crit pts we have already found first
            missing = 100-len(found_crit_pts)
            more_pts = np.random.normal(0, 1e3, size=(missing,dimInput,))
            points = np.concatenate((points, more_pts), axis=0)

            lr = 5 #5 #10 ##CHANGED
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(points)
            for step in range(2000): #2000  ##CHANGED
                # Use JaX's built in optimizer to do this.
                # We want to adjust the LR so that we get a better solution
                # as we optimize. Probably there is a better way to do this,
                # but this seems to work just fine.

                # No queries involvd here.
                if step%1000 == 0:
                    lr *= .5
                    optimizer = optax.adam(lr)
                    opt_state = optimizer.init(points)
                if step%100 == 0:
                    ell = loss(points, row)
                    if ell < 1e-5:
                        break
                points, opt_state = update(step, opt_state, points, row)
            for point in points:
                # For each point, try to see where it actually is.

                # First, if optimization failed, then abort.
                if loss(point, row) > 1e-5: 
                    continue

                if LAYER > 0:
                    # If we're on a deeper layer, and if a prior layer is zero, then abort
                    if min(np.min(np.abs(x)) for x in get_hidden_layers(point,A,B)) < 1e-4:
                        print("is on prior")
                        continue
                    
                solution = do_better_sweep(model, offset=point,low=-stepsize, high=stepsize, dataset = dataset)
                if len(solution) == 0:
                    stepsize *= 1.1
                elif len(solution) > 1:
                    stepsize /= 2
                elif len(solution) == 1:
                    stepsize *= 0.98
                    potential_solution = solution[0]

                    hidden_layer = forward(np.array(potential_solution),A,B, with_relu=True)
                    crit_val = matmul(hidden_layer, known_A[:,row], known_B[row])

                    #print("Crit val: ",crit_val)
                    if np.max(np.abs(crit_val))<1e-5:
                        critical_points.append(potential_solution)
                    #else:
                        #print("Reject it.")

        print("Finished with a total of", len(critical_points), "critical points")
        return critical_points


    critical_points_list = []
    for _ in range(1):
        NUM = dimOfPrevLayer*2
        try:
            critical_points_list.extend(get_more_points(NUM,prior_found_crit_pts))
        except AcceptableFailure:
            continue

        critical_points = np.array(critical_points_list)

        hidden_layer = forward(np.array(critical_points),A,B, with_relu=True)

        crit_val_1 = matmul(hidden_layer, known_A[:,row], known_B[row])

        best = (None, 1e6)
        upto = 100#100

        for iteration in range(upto):
            if iteration%1000 == 0:
                print("ITERATION", iteration, "OF", upto)
            if iteration%2 == 0 or True:

                # Try 1000 times to make sure that we get at least one non-zero per axis
                for _ in range(1000):
                    if NUM + 2 > len(hidden_layer):
                        print(f"Warning: Requested sample size (NUM+2) exceeds the size of hidden_layer. Adjusting NUM: ", NUM+2, len(hidden_layer))
                        NUM = max(0, len(hidden_layer) - 2)  # Adjust NUM to fit the available size
                    randn = np.random.choice(len(hidden_layer), NUM+2, replace=False)
                    if np.all(np.any(hidden_layer[randn] != 0, axis=0)):
                        break

                hidden = hidden_layer[randn]
                soln,*rest = np.linalg.lstsq(hidden, np.ones(hidden.shape[0]))
            crit_val_2 = matmul(hidden_layer, soln, None)-1
            
            quality = np.median(np.abs(crit_val_2))

            if iteration%100 == 0:
                print('quality', quality, best[1])
            if quality < best[1] and not np.any(np.abs(soln) < 1e-5): # Checks that solution is viable
                best = (soln, quality)
                #print("Crit val 2: ",crit_val_2)

            if quality < 1e-10: break
            if quality < 1e-10 and iteration > 1e4: break
            if quality < 1e-8 and iteration > 1e5: break

        soln, _ = best

        print("Compare",
              np.median(np.abs(crit_val_1)),
              best[1])
        #print("Crit val 1: ",crit_val_1)
        if best[0] is None:
            soln = known_A[:, row]

        if np.all(np.abs(soln) > 1e-10):
            break

    print('soln',soln)
    
    if np.any(np.abs(soln) < 1e-10):
        print("THIS IS BAD. FIX ME NOW.")
        raise
    
    rescale = np.median(soln/known_A[:,row])
    soln[np.abs(soln) < 1e-10] = known_A[:,row][np.abs(soln) < 1e-10] * rescale


    
    if best[1] < np.mean(np.abs(crit_val_1)) or True:
        return soln, -1 ### Recalculate bias?

def improve_layer_precision(LAYER, A, B, known_A, known_B,model,dimInput,dimOfPrevLayer,dimOfLayer, special,dataset,critical_groups):
    new_A = []
    new_B = []    
    try:
        if dataset == 'mnist': # This is the updated version which works for mnist but is not so good for the random models, also please optimise critical point search more if needed... Its not great right now
            out = map(improve_row_precision_ours,
                [(LAYER, A, B, known_A, known_B, row, model, dimInput, dimOfPrevLayer, dimOfLayer,special, dataset, False,critical_groups[row]) for row in range(dimOfLayer)])
        else:
            out = map(improve_row_precision_Carlini,
                [(LAYER, A, B, known_A, known_B, row, model, dimInput, dimOfPrevLayer, dimOfLayer,special, dataset, False,critical_groups[row]) for row in range(dimOfLayer)])
    except:
        raise
    new_A, new_B = zip(*out)

    new_A = np.array(new_A).T
    #Recalculate bias:
    point_per_class = [x[0] for x in critical_groups]
    # Compute the bias because we know wx+b=0
    extracted_bias = [matmul(forward(point_per_class[i],A,B, with_relu=True), new_A[:,i], c=None) for i in range(dimOfLayer)]

    # Don't forget to negate it.
    # That's important.
    # No, I definitely didn't forget this line the first time around.
    new_B = -np.array(extracted_bias)
    new_B = np.array(new_B)

    return new_A, new_B



