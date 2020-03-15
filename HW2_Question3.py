# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from math import log, sqrt

import numpy as np
import pandas as pd

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

message = [line.rstrip() for line in open("SMSSpamCollection")]

# for message_no,message in enumerate(message[:10]):
#     print(message_no,message)
#     print('\n')

    
message = pd.read_csv("SMSSpamCollection", sep = '\t', names = ["labes", "message"])
# print(message.head())

test_data = []
training_data = []
for i in range(10):
    if(i % 2 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])


print(test_data[0], "\n\n\n")
print(training_data[0])

