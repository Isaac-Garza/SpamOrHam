import numpy as np
import pandas as pd
import sklearn

message = [line.rstrip() for line in open("SMSSpamCollection")]

# for message_no,message in enumerate(message[:10]):
#     print(message_no,message)
#     print('\n')

message = pd.read_csv("SMSSpamCollection", sep = '\t', names = ["labes", "message"])
# print(message.head())

test_data = []
training_data = []
for i in range(150):
    if(i % 10 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])



# Helpful Links:
# https://stackoverflow.com/questions/47298070/importerror-no-module-named-wordcloud/53696236



