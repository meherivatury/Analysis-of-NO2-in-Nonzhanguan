import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
df = pd.read_csv('/Users/meherivatury/Desktop/Senior Year/Stat 443/Consulting Project/PRSA_Data_Nongzhanguan_20130301-20170228 .csv')
df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')
c = df.drop('No', axis=1)
c = c.drop('wd', axis=1)

#Encodes 'wd' as an integer
text = np.array(df['wd'])
label_encoder = LabelEncoder()
encoded_docs = label_encoder.fit_transform(text)
to_categorical(encoded_docs)
df['WDirection'] = encoded_docs

c = c['hour'].between(0, 17, inclusive = True)
df2 = df[c]
df2 = df2.drop('No', axis=1)
df2 = df2.drop('station', axis=1)
df2 = df2.drop('wd', axis=1)

d = []
def day_column(d):
    day = 0
    for i in range(0, 1461):
        day = day + 1
        for j in range(0, 18):
            d.append(day)
    return d
d = np.asarray(day_column(d))
df2['DayCount'] = d
AvgNO2 = df2.groupby('DayCount').agg(np.max)['NO2']

def classify(df):
    arr = ['Yellow']
    for i in range(0, 1460):
        if df.iat[i + 1] <= 40:
            arr.append('Green')
        elif df.iat[i + 1] > 40 and df.iat[i + 1] <= 80:
            arr.append('Yellow')
        elif df.iat[i + 1] > 80 and df.iat[i + 1] <= 120 :
            arr.append('Red')
        elif df.iat[i + 1] > 120:
            arr.append('Black')
    arr = np.asarray(arr)
    return arr
BucketedMAX = classify(AvgNO2)
corrected_data = df2.groupby('DayCount').agg(np.mean)
corrected_data['MaxNO2Color'] = BucketedMAX

corrected_data.to_csv('/Users/meherivatury/Desktop/Senior Year/Stat 443/Consulting Project/Corrected_Data_PRSA_Data_Nongzhanguan.csv', encoding='utf-8')

