import tensorflow as tf
import numpy as np
import math
from LoadData import LoadDataset
from HelperFunctions import *
import matplotlib.pyplot as plt

# Number of features per frame
FRAME_SIZE = 39
# Number of classes
NUM_CLASSES = 4

# Number of nodes per layer
HIDDEN_UNITS_1 = 100
HIDDEN_UNITS_2 = 50

# Parameters for the neural network
EPOCHS = 10
MINIBATCH_SIZE = 32
LEARNING_RATE = 0.004

# Loading the training data and splitting it into training and validation sets
DataX, DataY, meanArray, varianceArray = LoadDataset("TrainingData", 4, normalize = True, save = True)
train_size = int(0.95 * DataX.shape[0])

trainX = DataX[:train_size,:].T
trainY = DataY[:train_size,:].T
validX = DataX[train_size:,:].T
validY = DataY[train_size:,:].T

# Loading the test data and preprocessing it
testX, testY, _, __ = LoadDataset("TestData", 4, normalize = False, save = True)
testX = (testX-meanArray)/varianceArray
testX = testX.T
testY = testY.T

# Displaying the sizes of the datasets
print("The shape of training X is: " + str(trainX.shape))
print("The shape of training Y is: " + str(trainY.shape))
print("The shape of validation X is: " + str(validX.shape))
print("The shape of validation Y is: " + str(validY.shape))
print("The shape of test X is: " + str(testX.shape))
print("The shape of test Y is: " + str(testY.shape))

# Dividing the data into batches
minibatches = mini_batches(trainX, trainY, MINIBATCH_SIZE)
costs = []

with tf.Graph().as_default():

    # Creating placeholders for input
    frames_placeholder, labels_placeholder = placeholder_inputs(FRAME_SIZE, NUM_CLASSES)

    # Forward propagation
    logits = inference(frames_placeholder, HIDDEN_UNITS_1, HIDDEN_UNITS_2, FRAME_SIZE, NUM_CLASSES)

    # Calculating the cost
    cost = cost(logits,labels_placeholder)

    # Training using adam optimizer
    train_op = training(cost, LEARNING_RATE)

    # Evaluating predictions
    eval_correct = evaluation(logits,labels_placeholder)

    # Initializing variables
    init = tf.global_variables_initializer()

    sess = tf.Session()

    # Running the initializing operation
    sess.run(init)

    for i in range(EPOCHS * len(minibatches)):

        # Selecting a minibatch and splitting it into frames and labels
        minibatch = minibatches[i%len(minibatches)]
        (minibatch_X, minibatch_Y) = minibatch

        # Storing frames and labels in feed_dict to pass it to the placeholders
        feed_dict = {frames_placeholder: minibatch_X, labels_placeholder: minibatch_Y}

        # Running the training and cost operation
        _, minibatch_cost = sess.run([train_op, cost],feed_dict=feed_dict)
        costs.append(minibatch_cost)

        # Display the training accuracy and the validation accuracy after every epoch
        if (i+1) % len(minibatches) == 0:
            print("Epoch: " + str(int((i+1)/len(minibatches))))
            print("Training Data Accuracy")
            do_eval(sess, eval_correct, frames_placeholder, labels_placeholder, trainX, trainY, MINIBATCH_SIZE)

            print("Validation Data Accuracy")
            do_eval(sess, eval_correct, frames_placeholder, labels_placeholder, validX, validY, MINIBATCH_SIZE)

            # To babysit the model and guide learning
            decision = input("What do you want to do? Press \"enter\" to continue\n")
            if decision == "break":
                break
            if decision == "test":
                do_eval(sess, eval_correct, frames_placeholder, labels_placeholder, testX, testY, MINIBATCH_SIZE)
            if decision == "learning_rate":
                LEARNING_RATE = float(input("Enter new learning rate: "))

    # Plot cost vs minibatch steps
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('No of mini batch steps')
    plt.title("Learning rate =" + str(LEARNING_RATE))
    plt.show()

    # Display the test accuracy
    do_eval(sess, eval_correct, frames_placeholder, labels_placeholder, testX, testY, MINIBATCH_SIZE)

    # To save the model after training
    SaveDecision = input("Do you want to save the weights, mean array and variance array? Type \"Yes\" to save\n")
    if SaveDecision == "Yes":
        saver = tf.train.Saver(max_to_keep=1) 
        savePath = saver.save(sess, 'Models/MuClaEx.ckpt')