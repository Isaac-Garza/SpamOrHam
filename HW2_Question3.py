import numpy as np
import pandas as pd
import string

# Step 1
message = [line.rstrip() for line in open("SMSSpamCollection")]

# for message_no,message in enumerate(message[:10]):
#     print(message_no,message)
#     print('\n')

message = pd.read_csv("SMSSpamCollection", sep = '\t', names = ["labels", "message"])
# print(message.head())

test_data = []
training_data = []
print("Length of the message: ",len(message))
for i in range(10):
    if(i % 10 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])
print("-------------------Message DataFrame----------------------")
# print(message)

pd_test_data = pd.DataFrame(test_data)
pd_training_data = pd.DataFrame(training_data)

print("-------------------Test Data DataFrame:------------------- \n", pd_test_data)
print("-------------------Training Data DataFrame:--------------- \n", pd_training_data)


# Step 2 Remove Punctuation
print(pd_training_data.iloc[0])
mess = str(pd_training_data.iloc[0]['message'])
nopunc=[char for char in mess if char not in string.punctuation]
nopunc=''.join(nopunc)
print(nopunc)
pd_training_data.iloc[0]['message'] = nopunc
print("This is training mod: \n\n",pd_training_data.iloc[0]['message'])





# Helpful Links:
# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier



