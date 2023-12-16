import numpy as np
import librosa
import cv2
from keras.models import load_model

def get_genre_label(audio_file):

    preds = []
    labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

    song, rate = librosa.load(audio_file)

    song_length = len(song) // rate
    no_of_splits = song_length // 30

    song_splits = []
    for i in range(no_of_splits):
        split = song[i*rate*30:((i+1)*rate*30)]
        song_splits.append(split)

    model1 = load_model("models/new_spec_model_spectrogram1.h5")
    model2 = load_model("models/new_spec_model_spectrogram2.h5")
    model3 = load_model("models/new_ensemble_mfcc1.h5")
    model4 = load_model("models/new_ensemble_mfcc2.h5")
    model5 = load_model("models/new_ensemble_mfcc3.h5")
    model6 = load_model("models/model_melspectrogram1.h5")

    models = [model1, model2, model3, model4, model5, model6]
    
    def extract_features(song_split):
        spec_max = 52.859596
        mfcc_mean = -0.11748167
        mfcc_std = 58.0979
        mel_max = 40.267635

        song_stft = librosa.stft(song_split)
        song_stft = librosa.amplitude_to_db(abs(song_stft))
        song_stft = np.column_stack((song_stft,np.zeros(1025)))
        song_stft = song_stft/spec_max
        song_stft = song_stft.astype(np.float32)
        song_stft = song_stft.reshape((1,1025,1293,1))
    #     print(song_stft.shape)

        song_mel = librosa.feature.melspectrogram(y=song_split)
        song_mel = librosa.power_to_db(song_mel)
        song_mel = np.column_stack((song_mel,np.zeros(128)))
        song_mel = song_mel/mel_max
        song_mel = song_mel.astype(np.float32)
        song_mel = song_mel.reshape((1,128,1293,1))
    #     print(song_mel.shape)

        song_mfcc = librosa.feature.mfcc(y=song_split, sr=rate, n_mfcc=10)
        song_mfcc = np.column_stack((song_mfcc,np.zeros(10)))
        new = cv2.resize(song_mfcc,(600,120))
        new = new.astype(np.float32)
        song_mfcc = new.reshape((1,120,600,1))
        song_mfcc = (song_mfcc - mfcc_mean)/mfcc_std
        song_mfcc.shape
    #     print(song_mfcc.shape)
        
        return song_stft,song_mel,song_mfcc

    def get_majority(pred) :
        N = len(pred[0])
        vote = []
        for i in range(N) :
            candidates = [x[i] for x in pred]
            candidates = np.array(candidates)
            uniq, freq = np.unique(candidates, return_counts= True)
            vote.append(uniq[np.argmax(freq)])

        vote = np.array(vote)

        return vote
    
    def get_pred(song_stft,song_mel,song_mfcc,models):
        # Spectrogram model 1
        y_pred1 = models[0].predict(song_stft)
        y_pred1 = list(np.argmax(y_pred1, axis= -1))

        # Spectrogram model 2
        y_pred2 = models[1].predict(song_stft)
        y_pred2 = list(np.argmax(y_pred2, axis= -1))

        # MFCC model 1
        y_pred3 = models[2].predict(song_mfcc)
        y_pred3 = list(np.argmax(y_pred3, axis= -1))

        # MFCC model 2
        y_pred4 = models[3].predict(song_mfcc)
        y_pred4 = list(np.argmax(y_pred4, axis= -1))

        # MFCC model 3
        y_pred5 = models[4].predict(song_mfcc)
        y_pred5 = list(np.argmax(y_pred5, axis= -1))

        # Mel-spectrogram 
        y_pred6 = models[5].predict(song_mel)
        y_pred6 = list(np.argmax(y_pred6, axis= -1))

        # Get majority vote
        y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
        
        return y_pred
        

    
    for song_split in song_splits:
        song_stft, song_mel, song_mfcc = extract_features(song_split)
        preds = preds + get_pred(song_stft,song_mel,song_mfcc,models)
    
    final_pred = get_majority(preds)
    label = labels[final_pred[0]]

    return label

print(get_genre_label('Ed Sheeran - Shape of You.wav'))