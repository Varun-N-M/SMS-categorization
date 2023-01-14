import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

df = pd.read_table('SMSSpamCollection',names=['Labels','Messages'])
print(df.head().to_string())
print(df.info())

print(df['Labels'].value_counts())
le = LabelEncoder()
df['Labels'] = le.fit_transform(df['Labels'])

x = df['Messages']
y = df['Labels']

cv = CountVectorizer()
cv.fit(x)

unique_words = cv.get_feature_names_out()
print(len(unique_words))

word_array = cv.transform(x).toarray()

data_with_words = pd.DataFrame(data=word_array,columns=unique_words)

x_train,x_test,y_train,y_test = train_test_split(data_with_words,y,test_size=0.3,random_state=42)

model = MultinomialNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)*100

print(round(acc))
print(round(model.score(x_train,y_train)*100))
print(round(model.score(x_test,y_test)*100))

