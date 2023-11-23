# -------------------------------------------------------------------------
# AUTHOR: Suhuan Pan
# FILENAME: title of the source file
# SPECIFICATION: create and train neural networks with tensorflow and keras
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.


# importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------- Step 1: Using Keras to Load the Dataset --------
# Every image is represented as a 28×28 array rather than a 1D array of size 784.
# Moreover, the pixel intensities are represented as integers (from
# 0 to 255) rather than floats (from 0.0 to 255.0).
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# -------- creating a validation set and scaling the features --------
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000 :] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000 :]

n_X_train = len(X_train)
# print("Number of training set =", n_X_train)
# print("Number of y_test = ", len(y_test))
# print("number of y_valid = ", len(y_valid))

# For Fashion MNIST, we need the list of class names to know what we are dealing with.
# For instance, class_names[y_train[0]] = 'Coat'
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
n_outputLayer = len(class_names)
print("Number of output layer is number of class =", n_outputLayer)


# ----------------- build a model ------------------------*/
# specify the number of hidden layers and neurons in each layer.
def build_model(num_hidden_layers, num_neurons_hidden, num_neurons_output, learning_rate) :
    # Creating the Neural Network using the Sequential API
    #  model = keras.models.Sequential()
    model = keras.models.Sequential()

    # ---- input Layer with dimension 28 by 28----
    model.add(keras.layers.Flatten(input_shape = [28, 28]))

    # ---- add hidden layer based on ReLU activation function ----
    # iterate over the number of hidden layers to create the hidden layers:
    for i in range(num_hidden_layers) :
        model.add(keras.layers.Dense(num_neurons_hidden, activation = 'relu'))

    # ---- add output layer based on the softmax activation function  ----
    # one neural for each class since the classes are exclusive
    model.add(keras.layers.Dense(num_neurons_output, activation = "softmax"))

    # ---- defining the learning rate ----
    opt = keras.optimizers.SGD(learning_rate = learning_rate)

    # ---- Compiling the Model ----
    # specifying the loss function and the optimizer to use.
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

    return model


# ----------------- end of build a model ------------------------*/


# ----------------- Step 0: install tensorflow on terminal ------------------------*/
# source /Users/pan/Desktop/ML-DeepLearning/ML-DeepLearning/venv/bin/activate
# python -m pip install --upgrade tensorflow

# python -m pip install tensorflow-datasets
# python -m pip install tensorflow-models

# /Users/pan/Desktop/ML-DeepLearning/ML-DeepLearning/venv/lib/python3.7/site-packages/keras/datasets/fashion_mnist.py
# load data method: base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
# # change the base to the original data set
# # base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

# check for tensorflow GPU access and version
# print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
# print(f"TensorFlow version: {tf.__version__}")


# ----------------- hyper_parameters -----------------*/
# Iterate here over number of hidden layers,
# number of neurons in each hidden layer
# the learning rate.
n_hidden_layer_list = [2, 5, 10]
n_neurons_per_hidden_list = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]

highest_accuracy = 0.000
model = None
history = None
h = 0
n = 0
l = 0.000

#  ----------------- Iterate through different combinations of hidden layers and neurons
# looking or the best parameters w.r.t the number of hidden layers
for i1 in n_hidden_layer_list :
    # looking or the best parameters w.r.t the number of neurons
    for i2 in n_neurons_per_hidden_list :
        # looking or the best parameters w.r.t the learning rate
        for i3 in l_rate :
            # build the model for each combination by calling the function:
            # model = build_model(num_hidden_layers, num_neurons, input_dim=X_train.shape[1])
            model = build_model(i1, i2, n_outputLayer, i3)

            # Train the model on your training data
            # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            history = model.fit(X_train, y_train, epochs = n_outputLayer, validation_data = (X_valid, y_valid))
            # epochs = number times that the learning algorithm will work through the entire training dataset.

            # Evaluate the model on the validation set
            # Calculate the accuracy of this neural network
            # and store its value if it is the highest so far.
            # To make a prediction, do:
            class_predicted = np.argmax(model.predict(X_test), axis = -1)

            error = 0.00

            for k in range(len(class_predicted)) :
                if class_predicted[k] != y_test[k] :
                    error += 1

            accuracy = 1 - error / len(class_predicted)
            accuracy = round(accuracy, 4)

            # Keep track of the best model configuration
            if accuracy > highest_accuracy :
                highest_accuracy = accuracy
                model = history
                h = i1
                n = i2
                l = i3

            print("Highest accuracy so far: " + str(highest_accuracy))
            print("Parameters: ", "Number of Hidden Layers: ", str(h), ", number of neurons: ", str(n),
                  ", learning rate: ", str([l]))
            print()

# After generating all neural networks, print the summary of the best model found
# The model’s summary() method displays all the model’s layers, including each layer’s name (which is automatically
# generated unless you set it when creating the layer), its
# output shape (None means the batch size can be anything), and its number of parameters. Note that Dense layers
# often have a lot of parameters. This gives the model quite a lot of
# flexibility to fit the training data, but it also means that the model runs the risk of overfitting, especially
# when you do not have a lot of training data.

print(model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(model, to_file = img_file, show_shapes = True, show_layer_names = True)

# plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
