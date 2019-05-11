import librosa


audio_path = '../T08-violin.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))


print(x.shape, sr)