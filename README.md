# Music-Mood-Classifier
Breaks a song into frames, extracts relevant features and predicts the mood of each frame individually.

- Breaks the song into 0.64 second frames.
- Extracts the following features: RMS Energy, Spectral Centroid, Spectral Rolloff, Spectral Centroid, Spectral Bandwidth, MFFCs, ZCR, Tempo. These features essentially represent the intensity, timbre and rhythym features of each frame.
- Builds the training data from given clips that have been classified into 4 moods: Aggressive, Relaxed, Sad, Happy.
- Trains a neural network with the given clips.
- Takes a test song and builds the test set and outputs a .txt file with the mood of each 0.64 second frame of the song.

Given the subjective nature of moods, there is no metric for accuracy. The model depends strongly on the training data and how the user classifies the songs. 

NOTE: The project was a smaller part of a larger project. The project was to build a music visualizer that took into account the mood of the song and dynamically changed its visualizations to the song's mood. For doubts, contact me @ siddharth.nishtala(at)gmail(dot)com.
