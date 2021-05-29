import pandas as pd
import numpy as np
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer

df = pd.read_csv('/Users/meherivatury/Desktop/Senior Year/Stat 443/Consulting Project/Corrected_Data_PRSA_Data_Nongzhanguan.csv')
#Imputation
df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')

Y = df['MaxNO2Color']
Y = Y.ravel()

X = df.drop('MaxNO2Color', 1 )
X= X.drop('year', 1)
X = X.drop('hour', 1)
X = X.drop('DayCount', 1)
X = X.astype(float)


# encode wd values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():

    # create model
    model = Sequential()
    model.add(Dense(24, input_dim = 14, activation='relu'))
    model.add(Dense(20, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=30, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)) #prints k-fold cross validation accuracy as a percent


#Confusion table
y_pred = cross_val_predict(estimator, X, dummy_y, cv=10)
rounded_labels = np.argmax(dummy_y, axis= 1)
conf_mat = confusion_matrix(rounded_labels, y_pred)


