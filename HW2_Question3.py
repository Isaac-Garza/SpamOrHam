import numpy as np
import pandas as pd
import string

message = [line.rstrip() for line in open("SMSSpamCollection")]

# for message_no,message in enumerate(message[:10]):
#     print(message_no,message)
#     print('\n')

message = pd.read_csv("SMSSpamCollection", sep = '\t', names = ["labels", "message"])
# print(message.head())

test_data = []
training_data = []
for i in range(150):
    if(i % 10 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])

# testing punct. 
mess = 'sample message!...'
nopunc=[char for char in mess if char not in string.punctuation]
nopunc=''.join(nopunc)
print(nopunc)

# Helpful Links:
# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier



