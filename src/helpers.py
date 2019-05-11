import pandas as pd
from collections import Counter
import sys
sys.path.append('../speech-accent-recognition/src>')
import getsplit

from keras import utils
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10 #35#250

def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    # for index,language in enumerate(set(y)):
    #     print (index,language)
    lang_dict["Trinidad"] = 0
    lang_dict["Tobago"] = 1
    lang_dict["Barbados"] = 2

    y = list(map(lambda x: lang_dict[x],y))
    print (y)
    return utils.to_categorical(y, len(lang_dict))

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''

    y, sr = librosa.load('../Caribbean/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(normalize_mfcc(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)))
def to_mfcc_norm(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(len(wav) / chunk):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])

def normalize_mfcc(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)

