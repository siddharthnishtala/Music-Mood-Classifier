import tensorflow as tf
import numpy as np
import math

def mini_batches(X, Y, mini_batch_size = 64):
    '''Divide the dataset into minibatches'''
    m = X.shape[1]  
    mini_batches = []

    # Compute the number of complete datasets and divide them into batches
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Store the leftover examples in the final batch
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, (mini_batch_size * num_complete_minibatches):]
        mini_batch_Y = Y[:, (mini_batch_size * num_complete_minibatches):]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def mini_batches_feature_matrix(X, mini_batch_size = 64):
    '''Divide the feature matrix into minibatches'''
    m = X.shape[1]  
    mini_batches = []

    # Compute the number of complete datasets and divide them into batches
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        
        mini_batches.append(mini_batch_X)
    
    # Store the leftover examples in the final batch
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, (mini_batch_size * num_complete_minibatches):]

        mini_batches.append(mini_batch_X)
    
    return mini_batches

def placeholder_inputs(FRAME_SIZE, NUM_CLASSES):
    '''To create the placeholder for frames and labels'''
    train_frames = tf.placeholder(dtype = tf.float32, shape = (FRAME_SIZE, None))
    train_labels = tf.placeholder(dtype = tf.int64, shape = (NUM_CLASSES, None))

    return train_frames, train_labels

def frames_inputs(FRAME_SIZE, NUM_CLASSES):
    '''To create the placeholder for frames'''
    train_frames = tf.placeholder(dtype = tf.float32, shape = (FRAME_SIZE, None))

    return train_frames

def inference(frames, HIDDEN_UNITS_1, HIDDEN_UNITS_2, FRAME_SIZE, NUM_CLASSES):
    '''Forward propagation to compute logit scores'''
    # LAYER 1
    with tf.name_scope("Hidden_Layer_1"):
        with tf.variable_scope("Hidden_Layer_1"):
            weights = tf.get_variable("weights1", shape=[HIDDEN_UNITS_1, FRAME_SIZE],
                                initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            biases = tf.Variable(tf.zeros((HIDDEN_UNITS_1,1), dtype = tf.float32))
        hidden1 = tf.nn.relu(tf.matmul(weights,frames) + biases)

    # LAYER 2
    with tf.name_scope("Hidden_Layer_2"):
        with tf.variable_scope("Hidden_Layer_2"):
            weights = tf.get_variable("weights2", shape=[HIDDEN_UNITS_2, HIDDEN_UNITS_1],
                                initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            biases = tf.Variable(tf.zeros((HIDDEN_UNITS_2,1), dtype = tf.float32))
        hidden2 = tf.nn.relu(tf.matmul(weights,hidden1) + biases)

    # OUTPUT LAYER (without softmax)
    with tf.name_scope("Softmax_Layer"):
        with tf.variable_scope("Softmax_Layer"):
            weights = tf.get_variable("weights3", shape=[NUM_CLASSES, HIDDEN_UNITS_2],
                                initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            biases = tf.Variable(tf.zeros((NUM_CLASSES,1), dtype = tf.float32))
        logits = tf.matmul(weights,hidden2) + biases

    return logits

def cost(logits, labels):
    '''To compute the cost usng softmax cross entropy'''
    labs = tf.transpose(labels)
    logs = tf.transpose(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels= labs, logits=logs, name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name = 'xentropy_mean')

    return cost

def training(cost, LEARNING_RATE):
    '''To train the network using adam optimizer'''
    tf.summary.scalar('cost', cost)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    '''To evaluate the results'''
    correct = tf.nn.in_top_k(tf.transpose(logits), tf.argmax(labels), 1)

    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, eval_correct, frames_placeholder, labels_placeholder, data_X, data_Y, mini_batch_size):
    '''To evaluate and print the results in detail'''
    true_count = 0
    minibatches = mini_batches(data_X, data_Y, mini_batch_size)

    for i in range(len(minibatches)):

        minibatch = minibatches[i]
        (mini_batch_X, mini_batch_Y) = minibatch

        feed_dict = {frames_placeholder: mini_batch_X, labels_placeholder: mini_batch_Y}

        true_count += sess.run(eval_correct, feed_dict = feed_dict)

    accuracy = float(true_count)/data_X.shape[1]
    print("Number of examples: " + str(data_X.shape[1]))
    print("Number of correct prediction: " + str(true_count))
    print("Accuracy: " + str(accuracy))