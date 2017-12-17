import sys
import os
from LoadSong import LoadSong
import numpy as np

def LoadDataset(folder, noOfSubfolders, normalize = False, save = False):
    '''Load the dataset from a numpy file if available, else compute and load'''

    try:
        # Try to load a file if it already exists
        dataset_X = np.load(folder + "X.npy")
        dataset_Y = np.load(folder + "Y.npy")
        if folder == "TrainingData":
            meanArray = np.load("meanArray.npy")
            varianceArray = np.load("varianceArray.npy")
            return dataset_X, dataset_Y, meanArray, varianceArray
        if folder == "TestData":
            return dataset_X, dataset_Y, 1, 1

    except:
        # If loading a precomputed dataset fails, compute the dataset from the data
        SongsMatrices = []
        LabelsMatrices = []

        # Iterate through each subfolder in the given folder
        dirList = os.listdir("./" + folder)
        for subfolder in dirList:
            if subfolder.endswith(".DS_Store") or subfolder.endswith(".txt"):
                continue
            else:
                subdirList = os.listdir("./" + folder + "/" + subfolder)
                # Counter to store the number fo loaded clips
                songCounter = 0
                # Iterate through each file in the subfolder
                for file in subdirList:
                    if file.endswith(".DS_Store"):
                        continue
                    else:
                        # Load the features from the song and concatenate with the features of the previous songs
                        if songCounter != 0:
                            SongMatrix = LoadSong("./" + folder + "/" + subfolder + "/" + file)
                            SongsMatrix = np.concatenate((SongMatrix,SongsMatrix), axis=0)
                        else:
                            SongsMatrix = LoadSong("./" + folder + "/" + subfolder + "/" + file)

                        songCounter = songCounter + 1
                        print("Successfully loaded the clip: " + file)

                print("Loaded " + str(songCounter) + " clips from " + subfolder)
                print("The shape of the matrix is: " + str(SongsMatrix.shape))

                # Make labels for each subfolder
                Labels = np.zeros((SongsMatrix.shape[0],noOfSubfolders),dtype=int)
                if subfolder == "Happy":
                    Labels[:,0] = 1
                elif subfolder == "Sad":
                    Labels[:,1] = 1
                elif subfolder == "Relaxed":
                    Labels[:,2] = 1
                else:
                    Labels[:,3] = 1

                # Add the matrix to the list of songs matrices
                SongsMatrices.append(SongsMatrix)
                LabelsMatrices.append(Labels)
        
        # Concatenate all the song matrices
        X = np.concatenate((SongsMatrices[0],SongsMatrices[1],SongsMatrices[2],SongsMatrices[3]),axis=0)
        Y = np.concatenate((LabelsMatrices[0],LabelsMatrices[1],LabelsMatrices[2],LabelsMatrices[3]),axis=0)

        # Preprocessing
        meanArray = np.mean(X,axis=0)
        varianceArray = np.var(X,axis=0)
        if normalize:
            X = (X-meanArray)/varianceArray

        # Save the data if the save parameter of the function is true
        if save:
            np.save(folder + "X", X)
            np.save(folder + "Y", Y)
            if folder == "TrainingData":
                np.save("meanArray", meanArray)
                np.save("varianceArray", varianceArray)

        return X, Y, meanArray, varianceArray