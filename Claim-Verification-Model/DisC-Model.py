from sentence_transformers import SentenceTransformer
def get_vectors(sentences):
    model_name = 'bert-base-nli-cls-token'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings



import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf
import pandas as pd

import csv
import json
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def pre_process(sentence):
    ##Replace brackets
    brackets = ['-LRB-', '-LSB-', '-RRB-', '-RSB-']
    for bracket in brackets:
        sentence = sentence.replace(bracket, " ")
    return sentence

def get_training_data(file_path):
    label_dict = {"Supported": 0, "Refuted": 1}

    X_all = []
    y_all = []

    with open(file_path, 'r') as fp:
        reader = csv.reader(fp)
        next(reader)  # skip header row
        for row in reader:
            claim = pre_process(row[0])
            top1 = pre_process(row[5])
            X = (claim, top1)
            y = label_dict[row[1]]
            X_all.append(X)
            y_all.append(y)
    return X_all, y_all

def transform_features(file_path):
    X_all, y_all = get_training_data(file_path)

    claims = [claim for (claim, _) in X_all]
    evidences = [top1 for (_, top1) in X_all]
    print("Transforming claims")
    u = get_vectors(claims)
    print("Transforming evidences")
    v = get_vectors(evidences)
    print("Shape of u={} Shape of v={}".format(u.shape, v.shape))
    uplusv = u + v
    uminusv = u - v
    ubyv = u * v
    print("Resulting shapes=", uplusv.shape, uminusv.shape, ubyv.shape)
    all_features = np.concatenate((u, v, uplusv, uminusv, ubyv), axis=1)
    print("Shape of all=", all_features.shape)

    return all_features, y_all

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
name = "Neural Network(Simple Layer)"
model = Sequential()

def fit_predict(X_train, y_train, X_test, y_test, model_name="default", cr=True, cm=False):
    OPTIMIZER = RMSprop()
    DP = 0.2
    EPOCHS = 10
    VERBOSE = 1

    num_classes = len(set(y_train))
    if num_classes == 2:
        binary = True
        loss = 'binary_crossentropy'
    else:
        binary = False
        loss = 'categorical_crossentropy'

    model.add(Dense(300, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(DP))
    if not binary:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    model.compile(loss=loss, optimizer=OPTIMIZER, metrics=['accuracy'])

    if not binary:
        y_train = [label_to_vector(y, num_classes) for y in y_train]

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=1024, verbose=VERBOSE, validation_split=0.2)
    model.summary()
    y_pred = model.predict(X_test)
    if not binary:
        y_pred = [vector_to_label(vector) for vector in y_pred]
    else:
        y_pred = [int(round(i[0])) for i in y_pred]

    if cm:
        print(confusion_matrix(y_test, y_pred))
    if cr:
        print(classification_report(y_test, y_pred, digits=4))

    model.save(model_name)
    return y_pred

def label_to_vector(y, num_classes):
    vector = [0 for i in range(num_classes)]
    vector[int(y)] = 1
    return vector

def vector_to_label(vector):
    vector = vector.tolist()
    return vector.index(max(vector))

def predict(X_test):
    y_pred = model.predict(X_test)
    y_pred = [vector_to_label(vector) for vector in y_pred]
    return y_pred

def train_model(train_file, test_file):
    all_features_train, y_all_train = transform_features(train_file)
    all_features_test, y_all_test = transform_features(test_file)

    print("#Train=", len(all_features_train), len(y_all_train))
    print("#Test=", len(all_features_test), len(y_all_test))
    classifier_model_name = "baseline_classifier_inferbert_model.keras"
    fit_predict(all_features_train, y_all_train, all_features_test, y_all_test, classifier_model_name)

def train_infer_bert():
    train_file = ".../train.csv"
    test_file = ".../test.csv"
    train_model(train_file, test_file)

train_infer_bert()
