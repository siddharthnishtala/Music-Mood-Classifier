import tensorflow as tf
import sys
from LoadSong import *
from HelperFunctions import *

# Number of features per frame
FRAME_SIZE = 39
# Number of classes
NUM_CLASSES = 4

# Number of nodes per layer
HIDDEN_UNITS_1 = 100
HIDDEN_UNITS_2 = 50

# Loading the mean and variance arrays for preprocessing
try:
    meanArray = np.load("meanArray.npy")
    varianceArray = np.load("varianceArray.npy")
except:
    print("Cannot load mean array and variance array.")
    sys.exit(1)

# Taking the song name as a command line argument
try:
    SongName = sys.argv[1]
except:
    print("Please enter a song.\n")
    sys.exit(1)

# Loading the features of the song
try:
    FeatureMatrix = LoadSong(SongName)
except:
    print("File could not be loaded.")
    sys.exit(1)

# Preprocessing
FeatureMatrix = (FeatureMatrix-meanArray)/varianceArray

# minibatches = mini_batches_feature_matrix(FeatureMatrix.T, 128)

with tf.Session() as sess:

    # Restoring the trained model
    saver = tf.train.import_meta_graph('Models/MuClaEx.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Models/'))

    # Creating placeholders for input
    frames_placeholder = frames_inputs(FRAME_SIZE, NUM_CLASSES)

    # Loading the weights from the trained model
    graph = tf.get_default_graph()

    weights1 = graph.get_tensor_by_name("Hidden_Layer_1/weights1:0")
    biases1 = graph.get_tensor_by_name("Hidden_Layer_1/Hidden_Layer_1/Variable:0")
    weights2 = graph.get_tensor_by_name("Hidden_Layer_2/weights2:0")
    biases2 = graph.get_tensor_by_name("Hidden_Layer_2/Hidden_Layer_2/Variable:0")   
    weights3 = graph.get_tensor_by_name("Softmax_Layer/weights3:0")
    biases3 = graph.get_tensor_by_name("Softmax_Layer/Softmax_Layer/Variable:0")

    # Forward Propagation
    hidden1 = tf.nn.relu(tf.matmul(weights1,frames_placeholder) + biases1)
    hidden2 = tf.nn.relu(tf.matmul(weights2,hidden1) + biases2)
    logits = tf.matmul(weights3,hidden2) + biases3

    softmax_scores = tf.nn.softmax(logits, dim=0)
    labels = tf.add(tf.argmax(softmax_scores,axis=0,output_type=tf.int32),1)

    # Running the forward prop to get the scores and the labels
    scores, moods = sess.run([softmax_scores, labels], feed_dict={frames_placeholder:FeatureMatrix.T})

    # Saving the moods and scores as text files to pass to the visualizer
    np.savetxt("Moods.txt", moods, fmt='%1i')
    np.savetxt("Scores.txt", scores.T, fmt='%.2f')