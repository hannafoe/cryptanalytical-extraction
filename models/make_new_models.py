import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow_model_optimization as tfmot
import tempfile
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.datasets import mnist

def make_new_mnist_model(hidden_size, layer_number):
    # Load MNIST dataset
    tf.random.set_seed(42)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1 and flatten the input
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    # Define the model architecture
    input_layer = Input(shape=(784,), name='input')
    x = Dense(hidden_size, activation='relu', name='layer0')(input_layer)
    for i in range(1, layer_number):
        x = Dense(hidden_size, activation='relu', name=f"layer{i}")(x)
    # Output layer for multi-class as regression (non-standard approach)
    #output_layer = Dense(10, activation='softmax', name='output')(x)
    output_layer = Dense(1, activation='linear', name='output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Using mean_squared_error as a loss for this regression-like approach
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',#loss='sparse_categorical_crossentropy',#loss='binary_crossentropy',#loss='mean_squared_error',
                  metrics=['mean_absolute_error'])#metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))  # 100 epochs

    # Predict the test set
    predictions = model.predict(x_test)

    # Round predictions to the nearest integer
    rounded_predictions = np.round(predictions).astype(int).flatten()

    # Calculate accuracy
    accuracy = np.mean(rounded_predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Model summary
    model.summary()

    # Saving model path modified for MNIST
    model_save_path = f"deti/models/mnist784_{hidden_size}x{layer_number}_1v2.keras"
    model.save(model_save_path)
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            print("Weight: ",w)
            print("Bias: ", b)
    return model_save_path

#make_new_mnist_model(32,2)

def make_new_rescaled_model(model_path,scale):
    model = tf.keras.models.load_model(model_path)
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            w*=scale
            b*=scale
            l.set_weights([w,b])
    model_name_split = model_path.split('.')
    rescaled_model_path = model_name_split[0]+f"_rescaled_{scale}."+model_name_split[1]
    model.save(rescaled_model_path)


def make_new_random_model_carlini(input_size, hidden_size, number_layers, random_seed):
    # Seed the training model
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Randomly select sizes for input layer and hidden layer
    hidden_layers_sizes = [hidden_size] * number_layers
    output_size = 1

    # Generating random data for demonstration
    SAMPLES = 20
    X = np.random.normal(size=(SAMPLES, input_size))
    Y = np.array(np.random.normal(size=SAMPLES) > 0, dtype=np.float64).reshape(-1, 1)

    # Define the model architecture dynamically
    input_layer = Input(shape=(input_size,), name='input_layer')
    x = input_layer
        
    # Dynamically add hidden layers based on hidden_layer_sizes
    for i, size in enumerate(hidden_layers_sizes):
        x = Dense(size, activation='relu', name=f'layer{i}')(x)

    # Output layer
    output_layer = Dense(output_size, activation='sigmoid', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='mean_squared_error', metrics=['accuracy'])

    # Training the model
    model.fit(X, Y, epochs=100, batch_size=4, verbose=1)

    # Print model summary to check architecture and parameter count
    model.summary()

    # Saving the model
    model_save_path = f'deti/models/{input_size}_{hidden_size}x{number_layers}_{output_size}_Seed{random_seed}.keras'
    model.save(model_save_path)

#make_new_random_model_carlini(784,16,8,42)

def make_new_cifar_model(hidden_size,layer_number):
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1 and flatten the input
    x_train = x_train.reshape(-1, 3072) / 255.0
    x_test = x_test.reshape(-1, 3072) / 255.0

    # Define the model architecture
    input_layer = Input(shape=(3072,), name='input_2')
    x = Dense(hidden_size, activation='relu', name='layer0')(input_layer)
    for i in range(1, layer_number):
        x = Dense(hidden_size, activation='relu', name=f"layer{i}")(x)
    # Output layer
    output_layer = Dense(10, activation='softmax', name='output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with SGD optimizer and a momentum of 0.9
    model.compile(optimizer=SGD(momentum=0.9),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))#100

    # Model summary
    model.summary()

    model_save_path = f"deti/models/cifar3072_{hidden_size}x{layer_number}_10.keras"
    model.save(model_save_path)
    return model_save_path

#make_new_cifar_model(2,8)

def make_new_pruned_cifar_model(hidden_size,layer_number):
    model_save_path = make_new_cifar_model(hidden_size,layer_number)
    model = tf.keras.models.load_model(model_save_path)

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1 and flatten the input
    x_train = x_train.reshape(-1, 3072) / 255.0
    x_test = x_test.reshape(-1, 3072) / 255.0

    #Pruned model
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 64
    epochs = 2
    validation_split = 0.1 # 10% of training set will be used for validation set. 

    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                                final_sparsity=0.70,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer=SGD(momentum=0.9),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    logdir = tempfile.mkdtemp()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                    callbacks=callbacks)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.compile(optimizer=SGD(momentum=0.9),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
    model_for_export.summary()
    model_save_path_pruned = f'deti/models/cifar3072_{layer_number}x{hidden_size}_10_pruned.keras'
    model_for_export.save(model_save_path_pruned)

    #Accuracy of models
    base_model = tf.keras.models.load_model(model_save_path)
    pruned_model = tf.keras.models.load_model(model_save_path_pruned)
    _, baseline_model_accuracy = base_model.evaluate(
        x_test, y_test, verbose=0)
    _, pruned_model_accuracy = pruned_model.evaluate(
        x_test, y_test, verbose=0)
    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', pruned_model_accuracy)