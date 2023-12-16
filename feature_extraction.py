import os
import numpy as np
import librosa
from tqdm import tqdm
import librosa.display
import cv2
from sklearn.model_selection import train_test_split
import tensorflow

def extract_features():
    dirname = "Data/genres_original"
    
    audio_paths = []
    audio_label = []
    # Print all the files in different directories
    for root, dirs, files in os.walk(dirname, topdown=False):
        for filenames in files:
            if filenames.find('.wav') != -1:
                audio_paths.append(os.path.join(root, filenames))
                filenames = filenames.split('.', 1)
                filenames = filenames[0]
                audio_label.append(filenames)
    audio_paths = np.array(audio_paths)
    audio_label = np.array(audio_label)
    
    AllSpec = np.empty([1000, 1025, 1293])
    AllMel = np.empty([1000, 128, 1293])
    AllMfcc = np.empty([1000, 10, 1293])
    
    count = 0
    # Create a list for the corrupt indices
    bad_index = []
    for i in tqdm(range(len(audio_paths))):
        try:

            path = audio_paths[i]
            y, sr = librosa.load(path)
            # For Spectrogram
            X = librosa.stft(y)
            Xdb = librosa.amplitude_to_db(abs(X))
            AllSpec[i] = Xdb

            # Mel-Spectrogram 
            M = librosa.feature.melspectrogram(y=y)
            M_db = librosa.power_to_db(M)
            AllMel[i] = M_db

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 10)
            AllMfcc[i] = mfcc
        except Exception as e:
            bad_index.append(i)
    
    # Delete the features at the corrupt indices
    AllSpec = np.delete(AllSpec, bad_index, 0)
    AllMel = np.delete(AllMel, bad_index, 0)
    AllMfcc = np.delete(AllMfcc, bad_index, 0)
    
    # Convert features to float32 
    AllSpec = AllSpec.astype(np.float32)
    AllMel = AllMel.astype(np.float32)
    AllMfcc = AllMfcc.astype(np.float32)
    
    # Delete audio labels at corrupt indices
    audio_label = np.delete(audio_label, bad_index)
    
    # Convert labels from string into numerical value
    audio_label[audio_label == 'blues'] = 0
    audio_label[audio_label == 'classical'] = 1
    audio_label[audio_label == 'country'] = 2
    audio_label[audio_label == 'disco'] = 3
    audio_label[audio_label == 'hiphop'] = 4
    audio_label[audio_label == 'jazz'] = 5
    audio_label[audio_label == 'metal'] = 6
    audio_label[audio_label == 'pop'] = 7
    audio_label[audio_label == 'reggae'] = 8
    audio_label[audio_label == 'rock'] = 9
    audio_label = [int(i) for i in audio_label]
    audio_label = np.array(audio_label)
    
    # Convert numerical data into categorical data
    y = tensorflow.keras.utils.to_categorical(audio_label,num_classes = 10, dtype ="int32")
    
    # Save the features and labels as a .npz file
    np.savez_compressed("MusicFeatures.npz", spec= AllSpec, mel= AllMel, mfcc= AllMfcc, target=y)
    

def create_train_test_data():
    f = np.load('/kaggle/working/MusicFeatures.npz')
    spec = f['spec']
    mfcc = f['mfcc']
    mel = f['mel']
    y = f['target']
    
    # split train-test data
    spec_train, spec_test, mfcc_train, mfcc_test, mel_train, mel_test, y_train, y_test = train_test_split(spec, mfcc, mel, y, test_size= 0.2)
    
    # Spectrogram
    maximum1 = np.amax(spec_train)
    spec_train = spec_train/np.amax(maximum1)
    spec_test = spec_test/np.amax(maximum1)

    spec_train = spec_train.astype(np.float32)
    spec_test = spec_test.astype(np.float32)

    N, row, col = spec_train.shape
    spec_train = spec_train.reshape((N, row, col, 1))

    N, row, col = spec_test.shape
    spec_test = spec_test.reshape((N, row, col, 1))
    
    # MFCC
    newtrain_mfcc = np.empty((mfcc_train.shape[0], 120, 600))
    newtest_mfcc = np.empty((mfcc_test.shape[0], 120, 600))

    for i in range(mfcc_train.shape[0]) :
        curr = mfcc_train[i]
        curr = cv2.resize(curr, (600, 120))
        newtrain_mfcc[i] = curr

    mfcc_train = newtrain_mfcc

    for i in range(mfcc_test.shape[0]) :
        curr = mfcc_test[i]
        curr = cv2.resize(curr, (600, 120))
        newtest_mfcc[i] = curr

    mfcc_test = newtest_mfcc
    
    mfcc_train = mfcc_train.astype(np.float32)
    mfcc_test = mfcc_test.astype(np.float32)

    N, row, col = mfcc_train.shape
    mfcc_train = mfcc_train.reshape((N, row, col, 1))

    N, row, col = mfcc_test.shape
    mfcc_test = mfcc_test.reshape((N, row, col, 1))

    mean_data = np.mean(mfcc_train)
    std_data = np.std(mfcc_train)

    mfcc_train = (mfcc_train - mean_data)/ std_data
    mfcc_test = (mfcc_test - mean_data)/ std_data
    
    # Mel-Spectrogram
    maximum = np.amax(mel_train)
    mel_train = mel_train/np.amax(maximum)
    mel_test = mel_test/np.amax(maximum)

    mel_train = mel_train.astype(np.float32)
    mel_test = mel_test.astype(np.float32)

    N, row, col = mel_train.shape
    mel_train = mel_train.reshape((N, row, col, 1))

    N, row, col = mel_test.shape
    mel_test = mel_test.reshape((N, row, col, 1))
    
    # Save Spectrogram train-test
    np.savez_compressed("new_spectrogram_train_test.npz", S_train= spec_train, S_test= spec_test, y_train = y_train, y_test= y_test)

    # Save MFCC train-test
    np.savez_compressed("new_mfcc_train_test.npz", mfcc_train= mfcc_train, mfcc_test= mfcc_test, y_train = y_train, y_test= y_test)

    # Save Mel-Spectrogram train-test
    np.savez_compressed("new_mel_train_test.npz", mel_train= mel_train, mel_test= mel_test, y_train = y_train, y_test= y_test)