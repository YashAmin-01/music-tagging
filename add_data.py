import librosa
from time import time
import numpy as np
import soundfile as sf

def add_new_song(audio_file, tag):
    song, rate = librosa.load(audio_file)
    song_length = len(song) // rate
    no_of_splits = song_length // 30
    song_splits = []
    for i in range(no_of_splits):
        split = song[i*rate*30:((i+1)*rate*30)]
        song_splits.append(split)
        
    for i in range(no_of_splits):
        uid = int(time() * 100000)
        sf.write(f'Data/genres_original/{tag}.{uid}.wav', np.array(song_splits[i]), rate, subtype='PCM_24')