import os, sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

def train_models():
    spec_file = np.load("new_spectrogram_train_test.npz")
    spec_train = spec_file['S_train']
    y_train = spec_file['y_train']

    mfcc_file = np.load("new_mfcc_train_test.npz")
    mfcc_train = mfcc_file['mfcc_train']
    y_train = mfcc_file['y_train']
    
    mel_file = np.load("new_mel_train_test.npz")
    mel_train = mel_file['mel_train']
    y_train = mel_file['y_train']
    

    def create_models():
        def get_spec_model():
            spec_model = Sequential()
            spec_model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= (1025,1293,1), padding= 'same'))
            spec_model.add(MaxPooling2D((4,4), padding= 'same'))
            spec_model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
            spec_model.add(MaxPooling2D((4,4), padding= 'same'))
            spec_model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
            spec_model.add(MaxPooling2D((4,4), padding= 'same'))
            spec_model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            spec_model.add(MaxPooling2D((4,4), padding= 'same'))
            spec_model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            spec_model.add(MaxPooling2D((4,4), padding= 'same'))
            spec_model.add(Flatten())
            spec_model.add(Dense(128, activation= 'relu'))
            spec_model.add(Dense(64, activation= 'relu'))
            spec_model.add(Dense(10, activation= 'softmax'))
            spec_model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
            
            return spec_model
        
        def get_mfcc_model():
            model = Sequential()
            model.add(Conv2D(16, (3,3), input_shape= (120,600,1), activation= 'tanh', padding= 'same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((4,6), padding= 'same'))
            model.add(Conv2D(32, (3,3), activation= 'tanh', padding= 'same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((4,6), padding= 'same'))
            model.add(Conv2D(64, (3,3), activation= 'tanh', padding= 'same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((4,6), padding= 'same'))
            model.add(Flatten())
            model.add(Dense(256, activation= 'tanh'))
            model.add(Dense(64, activation= 'tanh'))
            model.add(Dense(10, activation= 'softmax'))
            model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
            
            return model
        
        def get_mel_model():
            model = Sequential()
            model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= (128,1293,1), padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Flatten())
            model.add(Dense(64, activation= 'relu'))
            model.add(Dense(10, activation= 'softmax'))
            model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
            
            return model
        
        spec_model_1 = get_spec_model()
        spec_model_2 = get_spec_model()
        mfcc_model_1 = get_mfcc_model()
        mfcc_model_2 = get_mfcc_model()
        mfcc_model_3 = get_mfcc_model()
        mel_model_1 = get_mel_model()
        
        return spec_model_1, spec_model_2, mfcc_model_1, mfcc_model_2, mfcc_model_3, mel_model_1
    
    spec_model_1, spec_model_2, mfcc_model_1, mfcc_model_2, mfcc_model_3, mel_model_1 = create_models()
    
    spec_model_1.fit(spec_train, y_train, epochs= 100, batch_size= 32)
    spec_model_1.save("models/new_spec_model_spectrogram1.h5")
    
    spec_model_2.fit(spec_train, y_train, epochs= 100, batch_size= 32)
    spec_model_2.save("models/new_spec_model_spectrogram2.h5")
    
    mfcc_model_1.fit(mfcc_train, y_train, epochs= 30, batch_size= 32, verbose= 1)
    mfcc_model_1.save("models/new_ensemble_mfcc1.h5")
    
    mfcc_model_2.fit(mfcc_train, y_train, epochs= 30, batch_size= 32, verbose= 1)
    mfcc_model_2.save("models/new_ensemble_mfcc2.h5")
    
    mfcc_model_3.fit(mfcc_train, y_train, epochs= 30, batch_size= 32, verbose= 1)
    mfcc_model_3.save("models/new_ensemble_mfcc3.h5")
    
    mel_model_1.fit(mel_train, y_train, epochs= 200, batch_size= 32, verbose= 1)
    mel_model_1.save("models/model_melspectrogram1.h5")