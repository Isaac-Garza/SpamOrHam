import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

# Step 1
print("-------------------Message DataFrame----------------------")
message = pd.read_csv('SMSSpamCollection',sep='\t',names=["labels","message"])

# for message_no,message in enumerate(message[::]):
#     print(message_no,message)
#     print('\n')

test_data = []
training_data = []

print("Length of the message: ", len(message))

for i in range(len(message)):
    if(i % 10 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])

# print(message)

pd_test_data = pd.DataFrame(test_data)
pd_training_data = pd.DataFrame(training_data)

# print(len(test_data), "DATA")
# print(len(pd_training_data), "TRAINING")

# print("-------------------Test Data DataFrame:------------------- \n", pd_test_data)
# print("-------------------Training Data DataFrame:--------------- \n", pd_training_data)

# Step 2 

# Remove Punctuation for Training Set
for i in range(len(pd_training_data)):
    mess = str(pd_training_data.iloc[i]['message'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    pd_training_data.iloc[i]['message'] = nopunc

# Remove Punctuation for Testing Set
for i in range(len(pd_test_data)):
    mess = str(pd_test_data.iloc[i]['message'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    pd_test_data.iloc[i]['message'] = nopunc

# SPAM string
# spam = ' '.join(list(pd_training_data[pd_training_data['labels'] == 'spam']['message']))

# HAM string
ham_words = ' '.join(list(pd_training_data[pd_training_data['labels'] == 'ham']['message']))

# print (spam)
# print (ham)

# Dictionary Frequency Counter for Ham
from collections import Counter
ham_word_freq = Counter(ham_words.split())

alpha = 0.2
N = 20000


ham_word_count = len(ham_words)

print("Print Freq Message: ", len(ham_word_freq))
print("Print Ham List:", len(ham_words))

# Calculate the P(word|ham) for every word
for key in ham_word_freq.keys():  
        ham_word_freq[key] = (ham_word_freq[key] + alpha)/(ham_word_count + (N*alpha))

print(ham_word_freq)














# Helpful Links:
# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier



