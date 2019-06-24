'''
Much of the code is modified from
https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Flatten, Merge, AveragePooling1D, Dropout
import keras.callbacks
from keras import regularizers
import numpy as np
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers import Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
np.random.seed(1337)


def model_woc(train,
              valid,
              test,
              train_rep,
              valid_rep,
              test_rep,
              y_tr,
              y_val,
              y_test,
              max_features,
              nb_class,
              emb_size,
              seq_length,
              nb_epoch,
              batch_size,
              lr,
              model_path, weight_path, dr, idim):

    model_cont = Sequential()
    model_cont.add(Embedding(max_features, output_dim=emb_size, input_length=seq_length, dropout=0.75, init='glorot_uniform'))
    model_cont.add(AveragePooling1D(pool_length=model_cont.output_shape[1]))
    model_cont.add(Flatten())

    # syntactic features
    model_stylo = Sequential()
    model_stylo.add(Dense(2, input_shape=(idim,), init='glorot_uniform', activation='relu'))
    model_stylo.add(Dropout(0.75))

    model_merge = Sequential()
    model_merge.add(Merge([model_cont, model_stylo], mode="concat"))
    model_merge.add(Dropout(dr))
    model_merge.add(Dense(nb_class, init='glorot_uniform', activation='softmax', name="output_author", activity_regularizer=regularizers.activity_l1(0.0005)))

    adam = Adam(lr=lr)
    model_merge.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    checkPoint = keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    print ("Training the model...")
    model_merge.fit([train, train_rep], y_tr,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_data=([valid,valid_rep], y_val),
                  callbacks=[earlystop_cb, checkPoint], verbose=2)
    loss, acc = model_merge.evaluate([test, test_rep], y_test, verbose=2)
    loss_val, acc_val = model_merge.evaluate([valid, valid_rep], y_val, verbose=2)
    print("Evaluation on test data using direct model (not saved one)")
    print("validation", loss_val, acc_val)
    print("test", loss, acc)

    # serialize model to JSON
    model_json = model_merge.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    l, a = loaded_model.evaluate([test, test_rep], y_test, verbose=2)
    l_valid, a_valid = loaded_model.evaluate([valid, valid_rep], y_val, verbose=2)
    print ("Evaluation on the validation and test data using saved model")
    print("validation", l_valid, a_valid)
    print("test", l, a)
    return l_valid, a_valid, l, a


def model_wac(train_char,
              valid_char,
              test_char,
              train_wd,
              valid_wd,
              test_wd,
              train_rep,
              valid_rep,
              test_rep,
              y_tr,
              y_val,
              y_ts,
              max_features_char,
              max_features_wd,
              nb_class,
              emb_size,
              seq_length_word,
              seq_length_char,
              nb_epoch,
              batch_size,
              lr,
              model_path,
              weight_path,
              dr):
    modelw = Sequential()
    modelw.add(Embedding(max_features_wd, emb_size, input_length=seq_length_word, dropout=0.75, init='glorot_uniform'))
    modelw.add(AveragePooling1D(pool_length=modelw.output_shape[1]))
    modelw.add(Flatten())

    modelc = Sequential()
    modelc.add(Embedding(max_features_char, emb_size, input_length=seq_length_char, dropout=0.75, init='glorot_uniform'))
    modelc.add(AveragePooling1D(pool_length=modelc.output_shape[1]))
    modelc.add(Flatten())

    # syntactic features
    model_stylo = Sequential()
    model_stylo.add(Dense(2, input_dim=42, init='glorot_uniform', activation='relu'))
    model_stylo.add(Dropout(0.75))

    model_merge = Sequential()
    model_merge.add(Merge([modelw, modelc], mode='max'))

    model_out = Sequential()
    model_out.add(Merge([model_merge, model_stylo], mode='concat'))
    model_out.add(Dropout(dr))
    model_out.add(Dense(nb_class, init='glorot_uniform', activation='softmax', name="output_author", activity_regularizer=regularizers.activity_l1(0.0005)))

    adam = Adam(lr=lr)
    model_out.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    checkPoint = keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    print ("Training the model...")
    model_out.fit([train_wd, train_char, train_rep], y_tr,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_data=([valid_wd,valid_char, valid_rep], y_val),
                  callbacks=[earlystop_cb, checkPoint], verbose=2)

    loss, acc = model_out.evaluate([test_wd, test_char, test_rep], y_ts, verbose=2)
    loss_val, acc_val = model_out.evaluate([valid_wd, valid_char, valid_rep], y_val, verbose=2)
    print("Evaluation on test data using direct model (not saved one)")
    print("validation", loss_val, acc_val)
    print("test", loss, acc)

    # serialize model to JSON
    model_json = model_out.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    l, a = loaded_model.evaluate([test_wd, test_char, test_rep], y_ts, verbose=2)
    l_valid, a_valid = loaded_model.evaluate([valid_wd, valid_char, valid_rep], y_val, verbose=2)
    print ("Evaluation on the validation and test data using saved model")
    print("validation", l_valid, a_valid)
    print("test", l, a)
    return l_valid, a_valid, l, a


def model_dense(
              train_rep,
              valid_rep,
              test_rep,
              y_tr,
              y_val,
              y_test,
              nb_class,
              nb_epoch,
              batch_size,
              lr,
              model_path, weight_path, hidden_size, idim, dr):
    model_stylo = Sequential()
    model_stylo.add(Dense(hidden_size, input_shape=(idim,), init='glorot_uniform', activation='relu'))
    model_stylo.add(Dropout(dr))
    model_stylo.add(Dense(nb_class, init='glorot_uniform', activation='softmax', name="output_author", activity_regularizer=regularizers.activity_l1(0.0005)))
    print (model_stylo.summary())
    adam = Adam(lr=lr)
    model_stylo.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    checkPoint = keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    print ("Training the model...")
    model_stylo.fit(train_rep, y_tr,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_data=(valid_rep, y_val),
                  callbacks=[earlystop_cb, checkPoint], verbose=2)

    # serialize model to JSON
    model_json = model_stylo.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    l, a = loaded_model.evaluate(test_rep, y_test, verbose=2)
    l_valid, a_valid = loaded_model.evaluate(valid_rep, y_val, verbose=2)
    print ("Evaluation on the validation and test data using saved model")
    print("validation", l_valid, a_valid)
    print("test", l, a)
    return l_valid, a_valid, l, a

