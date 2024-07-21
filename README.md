# Parameter Recovery Attack on ReLU-based Deep Neural Networks

Implementation of an attack to recover the signatures and signs of a ReLU-based deep neural network (DNN). The code runs attacks for one layer of a DNN at each time.

# Explanation of this Codebase

This codebase bases its structure from Canales-Martinez et al's codebase that can be found under https://anonymous.4open.science/r/deti-C405. The codebase is a unification of code from Carlini et al.'s codebase (https://github.com/google-research/cryptanalytic-model-extraction) and Canales-Martinez et al's codebase. A lot of snippets of code are from their respective codebases. We use Carlini et al.'s signature extraction and Canales-Martinez et al's sign extraction as basis code. Carlini et al.'s functions have been slightly adapted to fit into the format of this new codebase, removing lots of global variable dependencies and cleaning away parts that were not needed. The extraction also now runs all in jax. Tensorflow is only used as an initial carrier for the weights and bias.

# Reproduce Attacks

To reproduce the results of the attacks reported in our manuscript, please execute the following commands: 
```
python -m neuronWiggle --model models/mnist784_16x2_1.keras --layerID 2 --seed 20 --dataset 'mnist' --quantized 2
```
```
python -m neuronWiggle --model models/50_25x2_1_Carlini.keras --layerID 2 --seed 20 --quantized 1
```
The `--seed` option allows to try extraction on different seeds. The `--quantized` options allow sign extraction in float16 (option 1), in float32 (option 2) or in float64 (option 0). The default is float32. For float64 the precision improvement function in the signature recovery is also run, whereas it is not run for float16 and float32. With `--signRecoveryMethod` we can signal if we want `carlini` or `neuronWiggle`, where default is `neuronWiggle`. With `--onlySign` set to True we can run only the sign recovery, skipping the signature recovery. The cifar models can only be run with this option because we have not implemented signature recovery for them.

# Create a new model

A new model can be created in the file `models\make_new_models.py`. The file can be run with the command 
```
python .\models\make_new_models.py
```
Prior to running this, however, the function that one wants to run needs to be commented back in in the `make_new_models.py` file. Specific comments for that are left under the functions to make new mnist models and new random models. In this, one can specify the number of hidden layers, the number of neurons for the hidden layers and the training seed.

# Dependencies

The code execution relies on the Python modules denoted below. The experiments were run on Python 3.10.11. In case the code does not run, here are the versions used:

```
pip install tensorflow==2.15.0
pip install numpy==1.25.2
pip install pandas==2.1.2
pip install jax==0.4.30
pip install optax==0.2
pip install scipy==1.9.3
```

# Detailed Description of Workings

## In `neuronWiggle.py`:

**Main:** The `__main__` function previouly in Canales-Martinez et al's codebase only ran the neuronwiggle sign recovery. Now it has been significantly adapted to include the signature recovery and quantization options and a way to test Carlini's sign recovery as well. 

**Sign extraction neuron wiggle:** The `recoverSign` function is Canales-Martinez's neuronwiggle sign recovery function. Some checks and metrics have been added, as well as the quantized options. The biggest change for runtime is the switch of prediction calls from model.predict() in tensorflow to a manual predict function using jax matrix multiplications of the input and weights and bias.

**Sign extraction Carlini:** The `recoverSign_Carlini` function is adapted from Carlini's `run_full_attack()` function and runs Carlini's exponential time sign extraction. Only some variables and metrics have been changed. `solve_final_layer` and `solve_second_from_final_layer` are also similarly from the `extract.py` file from Carlini and have only been slightly adapted. 

## In `blackbox_src`: 

### Critical Point Search

**Functions for the critical point search:** `sweep_for_critical_point` and `do_better_sweep` are originally from Carlini's `src/find_witnesses.py` file. They have been adapted to also handle MNIST models.

**Functions for the targeted critical point search which can be initiated if at least 3 critical point has been found for each neuron:** These functions are from from Carlini's `src/sign_recovery.py` file. `is_on_following_layer` is a function  which determines which layer a critical point is on. `binary_search_towards` is a function which determines how far one can walk along the hyperplane until it is in a different polytope from a prior layer. `find_plane_angle` is a function which determines which way the hyperplane will bend after the critical point. `follow_hyperplane` is the function which calls `binary_search_toward` and `find_plane_angle` which tries to find new critical points for a neuron. `get_more_crit_pts` is the parent function which calls `is_on_following_layer` and `follow_hyperplane` which checks if diversity in view has been achieved for a neuron so that a full signature can be recovered. These functions have been barely touched except some tweaks to identify the index in the full signature that is still missing.

### Signature Extraction

**Functions for clustering the partial signatures and critical points to belong to the same neuron:** `process_bock` is a function that computes the pairwise similarity between partial signatures. `ratio_normalize` is the function that computes the pairwise ratios between two partial signatures. `graph_solve` is the main function that computes the clusters. Parts have been changed here that prevent some infinite loop scenarios and errors from happening. Memory deduplication has also been added here.

**Functions for actual Signature recovery:** These functions can be found under `src/hyperplane_normal.py` and `src/layer_recovery` in Carlini's codebase. `get_ratios` is a helper function that computes the full signature of one neuron on the first layer. `get_ratios_lstsq` is the helper function that computes the partial signatures for one neuron in a later layer. `gather_ratios` is their parent function which navigate signature recovery for all critical points found. `compute_layer_values` is the main function of signature recovery that puts everything together. It outputs the full signature for all neurons in a layer if it completes successfully. This is also where the different failure cases are addressed with not having enough partial signatures to complete a full signature. Only minor changes have been made to implement metrics and memory deduplication.

**Functions for Precision Improvement:** `improve_row_precision_Carlini` is the row precision improvement function from Carlini and our adapted version for mnist functions is `improve_row_precision_ours`. `improve_layer_precision` brings everything together.

### Sign Extraction

**Functions for Sign Recovery from Canales-Martinez:** `getOrthogonalBasis`, `findCorner`, `getLocalMatrixAndBias`, `getHiddenVector`, `getOrthogonalBasisForInnerLayerSpace` are the main functions for Canales-Martinez's sign extraction and they have not been changed much. `findCorner_quantized` was added to deal with quantized sign extraction.

**Functions for Sign recovery from Carlini:** `solve_contractive_sign`, `is_solution_map`, `is_solution`, and `solve_layer_sign` are from Carlini's exponential time sign extraction. Only metrics and global variable dependencies have been changed.

**SOE Sign Recovery from Canales-Martinez:** Canales-Martinez et al. have another sign extraction method called soe sign recovery which is more efficient for layer 1 which is not included in this codebase since this was not focus of our work. However, please refer to it on their codebase (https://anonymous.4open.science/r/deti-C405) if interested.

### Util functions

**Util functions:** `AcceptableFailure` and `GatherMoreData(AcceptableFailure)` are from Carlini's `src/utils.py` file. `getCIFARtestImage` has been slightly adapted from Canales-Martinez to give batches of images and `getMNISTtestImage` was added to output test images similarly.

**Basic helper functions:** `basis`, `matmul`, `forward`, `forward_at`, `get_hidden_layers`, `get_hidden_at`, `get_polytope`, `get_polytope_at`, `get_grad`, `get_second_grad_unsigned`, `getAllWeightsAndBiases`, `predict_manual_fast`, `get_saved_queries`,`get_query_counts`,`softmax`. Most of these are from Carlini and some are our own additional helper functions.

**Functions for quality check:** `check_quality` is from Carlini's `src/utils.py` file to check the quality of extraction by scaling the extracted weights similarly to the original weights and comparing the similarity of weights.

# Further Note on Full Pipeline Recovery

The code only runs attacks for one layer of a DNN at each time. So, it is in a way an idealised environment where all previous layers are always assumed to be correct. To test the hypothesis on errors in the sample distance check, we changed the weights of the previous layers as can be seen in comments in the code. To extract a whole model from first to last layer the code can just be run for each layer separately. If we want a non-idealised full run then we can do so, since the extracted parameters are always saved instead of the original weights of the target layer as a new tensorflow model "{modelname}_extracted_{other_args}". Then, the next layer is still run with the original modelname as the argument but in the `neuronWiggle.py` `__main__` function there is a section marked quite at the top where signatures are recovered that denotes where to read in the extracted model instead of the original model (around line 462). This is how we would be able to run different sign combinations of the model and determine which combination was faulty. In the sign extraction there is an area marked which throws errors if the sample distance check is thrown more than 15 times and in the main method we count how often this happens and if it is more than 5 we terminate the whole execution and suggest to try with a different sign combination. If less than 5 signatures are incorrect then it is suggested to go back into the signature extraction and reextract the concerning signatures. (If we wanted to run a full attack in a non-idealised setting the code could obviously be changed in various parts to do this automatically but this was not the focus of our work.)