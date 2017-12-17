import librosa
import numpy as np

def LoadSong(path, WeightFeatures = True):
    '''To compute and load the features of the given clip'''

    # Load the file
    y, sr = librosa.load(path,sr=16000,mono=True,offset=10)
    y = y[:y.shape[0]-160000]

    # Divide the audio time series into 0.64 second frames
    noOfFrames = int(y.shape[0]/10240) - 1
    frames = np.ndarray(shape=(noOfFrames,10240),dtype=float)

    for i in range(noOfFrames):
        frames[i,:] = np.transpose(y[i*10240:(i+1)*10240])

    # Initialize all feature vectors
    RMSEvec = np.ndarray(shape=(noOfFrames,1),dtype=float)
    SpCenVec = np.ndarray(shape=(noOfFrames,3),dtype=float)
    SpCoVec = np.ndarray(shape=(noOfFrames,14),dtype=float)
    SpRoVec = np.ndarray(shape=(noOfFrames,3),dtype=float)
    SpBanVec = np.ndarray(shape=(noOfFrames,3),dtype=float)
    MFFCsVec = np.ndarray(shape=(noOfFrames,13),dtype=float)
    ZCRvec = np.ndarray(shape=(noOfFrames,1),dtype=float)
    TempoVec = np.ndarray(shape=(noOfFrames,1),dtype=float)
    
    for i in range(noOfFrames):
        # Compute the RMS energy
        RMSEvec[i] = librosa.feature.rmse(frames[i,:],frame_length=10240,center=False)

        # Compute the Spectral Centroid
        SpCenVec[i,:] = np.reshape(librosa.feature.spectral_centroid(frames[i,:],sr=16000,n_fft=10240,hop_length=5120),(3,))
        
        # Compute the Spectral Contrast 
        SpCoVec[i,:] = np.reshape(librosa.feature.spectral_contrast(frames[i,:],sr=16000,n_fft=10240,hop_length=10240),(14,))

        # Compute the Spectral Rolloff
        SpRoVec[i,:] = np.reshape(librosa.feature.spectral_rolloff(frames[i,:],sr=16000,n_fft=10240,hop_length=5120),(3,))

        # Compute the Spectral Bandwidth
        SpBanVec[i,:] = np.reshape(librosa.feature.spectral_bandwidth(frames[i,:],sr=16000,n_fft=10240,hop_length=5120),(3,))

        # Compute the MFFCs
        MFFCsVec[i,:] = np.reshape(librosa.feature.mfcc(frames[i,:],sr=16000,n_mfcc=13,n_fft=10240,hop_length=20480),(13,))

        # Compute the Zero Crossing Rate 
        ZCRvec[i,:] = librosa.feature.zero_crossing_rate(frames[i,:],frame_length=10240,center=False)

        # Compute the tempo
        TempoVec[i,:] = librosa.beat.tempo(y=frames[i,:], sr=22050, hop_length=20480, start_bpm=30, std_bpm=1.0, ac_size=8.0, max_tempo=500)

    # Concatenate all feature vectors into a feature matrix
    FeatureMatrix = np.concatenate((RMSEvec,SpCenVec,SpCoVec,SpRoVec,SpBanVec,MFFCsVec,ZCRvec,TempoVec),axis=1)

    if WeightFeatures:
        # To reduce the noise in data and to take the neighbouring frames into account
        FeatureMatrix = WAverageFeatures(FeatureMatrix)

    return FeatureMatrix

def WAverageFeatures(FeatureMatrix):
    '''To compute the weighted average of each frames with the neighbouring frames'''

    AveragedFeatureMatrix = np.ndarray(shape=FeatureMatrix.shape,dtype=float)
    noOfFrames = FeatureMatrix.shape[0]

    # Weight vector for the first 7 frames
    WeightVector1 = np.array([0.6, 0.2, 0.1, 0.05, 0.025])
    # Weight vector for the last 7 frames
    WeightVector2 = np.array([0.025, 0.05, 0.1, 0.2, 0.6])
    # Weight vector for the middle frames
    WeightVector3 = np.array([0.025, 0.05, 0.1, 0.6, 0.1, 0.05, 0.025])
    
    # Computing the weighted average and storing it
    for i in range(noOfFrames):
        if i < 7:
            AveragedFeatureMatrix[i,:] = np.matmul(WeightVector1, FeatureMatrix[i:i+5,:])
        elif i > noOfFrames - 7:
            AveragedFeatureMatrix[i,:] = np.matmul(WeightVector2, FeatureMatrix[i-4:i+1,:])
        else:
            AveragedFeatureMatrix[i,:] = np.matmul(WeightVector3, FeatureMatrix[i-3:i+4,:])

    return AveragedFeatureMatrix