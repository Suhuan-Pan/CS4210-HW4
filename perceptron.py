# -------------------------------------------------------------------------
# AUTHOR: Suhuan Pan
# FILENAME: ML-HW4/ML-HW4/perceptron.py
# SPECIFICATION: perceptron classifier and multiple layer classifier
# FOR: CS 4210- Assignment #4
# TIME SPENT: 5 hours
# -----------------------------------------------------------*/
import math

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# ----------------------- Load Data ----------------------------*/
df = pd.read_csv('optdigits.tra', sep = ',', header = None)  # reading the data by using Pandas library

X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:, -1]  # getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep = ',', header = None)  # reading the data by using Pandas library

X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:, -1]  # getting the last field to form the class label for test

# number of training samples determines the total level of layers

n_samples = X_training.shape[0]
n_features = X_training.shape[1]
n_output = y_test.shape[0]


pct_accuracy = 0.00
mlp_accuracy = 0.00

h11 = 0
h12 = 0
h21 = 0
h22 = 0

# iterates over set of different learning rate
for ni in range(len(n)):

    # iterates over shuffle is True or False
    for ri in range(len(r)):

        p_acc = 0.00
        m_acc = 0.00


        # iterates over both perceptron and MLP algorithms
        # 1. Create a Neural Network classifier
        # 2. Fit the Neural Network to the training data
        # 3. make the classifier prediction for each test sample
        # 4. start computing its accuracy
        # hint: to iterate over two collections simultaneously with zip() Example:

        for n_hidden_layer in range(2):

            e1 = 0
            e2 = 0

            if n_hidden_layer == 0:

                clf1 = Perceptron(max_iter = 1000, shuffle = r[ri], eta0 = n[ni])
                clf1.fit(X_training, y_training)
                y_predict1 = clf1.predict(X_test)

                for i in range(len(y_test)):
                    if y_predict1[i] != y_test[i]:
                        e1 += 1

                p_acc = round(1 - (e1 / n_samples), 2)
                if p_acc > pct_accuracy:
                    pct_accuracy = p_acc
                    h11 = ni
                    h12 = ri


            else:
                # FIXED: learning_rate is constant with its init by default
                clf2 = MLPClassifier(hidden_layer_sizes = (25,), activation = "logistic",
                                     learning_rate_init = n[ni],
                                     max_iter = 1000, shuffle = r[ri])

                clf2.fit(X_training, y_training)
                y_predict2 = clf2.predict(X_test)

                for i in range(len(y_test)):
                    if y_predict2[i] != y_test[i]:
                        e2 += 1

                m_acc = round(1 - (e2 / n_samples), 2)
                if m_acc > mlp_accuracy:
                    mlp_accuracy = m_acc
                    h21 = ni
                    h22 = ri

        # 5a. check if the calculated accuracy is higher than the previously one calculated for each classifier.
        # 5b. If so, update the highest accuracy
        # 5c. print it together with the network hyper_parameter
        print("Highest Perceptron accuracy so far: ", pct_accuracy)
        print("Parameters: learning rate =", n[h11], "shuffle =", r[h12])

        print("Highest MLP accuracy so far: ", mlp_accuracy)
        print("Parameters: learning rate is constant learn_rate_init =", n[h21], "shuffle =", r[h22])



print("Highest Perceptron accuracy: ", pct_accuracy,
      "Parameters: learning rate =", n[h11], "shuffle =", r[h12])

print("Highest MLP accuracy: ", mlp_accuracy,
      "Parameters: learning rate is constant learn_rate_init =", n[h21], "shuffle =", r[h22])
