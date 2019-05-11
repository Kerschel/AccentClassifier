import pandas as pd
from collections import Counter
import sys
sys.path.append('../speech-accent-recognition/src>')
import getsplit
import helpers as k
from keras import utils
import accuracy
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
import trainmodel
from sklearn.metrics import classification_report


EPOCHS =10

def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))

def train_model(X_train,y_train,X_validation,y_validation, batch_size=20): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])
    print("Number of classes:" ,num_classes)

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols)
    X_train = X_train.reshape(X_train.shape[0], rows, cols )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[1:], 'training samples')


    print(X_train[0].shape)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, stateful=False,activation ='relu'))
    model.add(Dropout(.20))
    model.add(LSTM(64, return_sequences=True, activation ='relu'))

    model.add(LSTM(64, return_sequences=True, stateful=False ,activation ='relu'))
    model.add(Dropout(.20))
    model.add(Flatten())

    model.add(Dense(10, activation='relu'))
    model.add(Dropout(.20))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # Fit model using ImageDataGenerator
    # model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
    #                     steps_per_epoch=len(X_train) / 32
    #                     , epochs=EPOCHS,
    #                     callbacks=[es,tb], validation_data=(X_validation,y_validation))
    print("X val:",X_validation.shape)
    print("y val:",y_validation.shape)
    print("X train:",X_train.shape)
    print("y train:",y_train.shape)
    model.fit(X_train,y_train,batch_size=batch_size,epochs=EPOCHS,validation_data=(X_validation,y_validation))

    return (model)

if __name__ == '__main__':
    file_name = sys.argv[1]
    # model_filename = sys.argv[2]

    # Load metadata
    df = pd.read_csv(file_name,encoding='ISO-8859-1')

    # print (df)
    # Filter metadata to retrieve only files desired
    filtered_df = getsplit.filter_df(df)

    X_train, X_test, y_train, y_test = getsplit.split_people(filtered_df)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(k.get_wav, X_train)
    X_test = pool.map(k.get_wav, X_test)

    X_train = pool.map(k.to_mfcc, X_train)
    X_test = pool.map(k.to_mfcc, X_test)
    X_train, y_train = k.make_segments(X_train, y_train)
    X_validation, y_validation = k.make_segments(X_test, y_test)


    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)
    print("Y train shape",np.array(y_train).shape)

    model = train_model(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation))

    
    # y_predicted=model.predict_classes(X_validation, batch_size=32)
    # print(classification_report(y_validation, y_pred))


    # # y_predicted = accuracy.predict_class_all(k.create_segmented_mfccs(X_test), model)

    # print('Confusion matrix of total samples:\n', np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
    # print('Confusion matrix:\n',accuracy.confusion_matrix(y_predicted, y_test))

    # print('Accuracy:', accuracy.get_accuracy(y_predicted,y_test))

  
  # # add dropout to control for overfitting
  # model.add(Dropout(.25))

  # # squash output onto number of classes in probability space




