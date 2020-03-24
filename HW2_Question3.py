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

# -----Training Data-----
# SPAM string
spam_words = ' '.join(list(pd_training_data[pd_training_data['labels'] == 'spam']['message']))

# HAM string
ham_words = ' '.join(list(pd_training_data[pd_training_data['labels'] == 'ham']['message']))

# Both Ham and Spam String
all_words = spam_words + ' ' + ham_words

# print (spam)
# print (ham)


from collections import Counter
# Dictionary Frequency Counter for Ham
ham_word_freq = Counter(ham_words.split())

# Dictionary Frequency Counter for Spam
spam_word_freq= Counter(spam_words.split())

# Dictionary Frequency Counter for all words
all_words_freq = Counter(all_words.split())


alpha = 0.2
N = 20000


total_freq_of_words = len(all_words_freq)

print("Print Freq Message:", len(ham_word_freq))
print("Print Spam Freq:", len(spam_words))
print("Print Ham Freq:", len(ham_words))

# Calculate the P(word|ham) for every word
for key in ham_word_freq.keys():  
    ham_word_freq[key] = (ham_word_freq[key] + alpha)/(total_freq_of_words + (N * alpha)) 

for key in spam_word_freq.keys():
    spam_word_freq[key] = (spam_word_freq[key] + alpha)/(total_freq_of_words + (N * alpha))

# print(ham_word_freq)
# print(spam_word_freq)

ham_data = pd.DataFrame.from_dict(ham_word_freq, orient='index').reset_index()
ham_data.rename(columns = {'index':'word', 0:'P(word|ham)'}, inplace = True)


spam_data = pd.DataFrame.from_dict(spam_word_freq, orient = 'index').reset_index()
ham_data.rename(columns = {'index':'word', 0:'P(word|spam)'}, inplace = True)


print(ham_data)
print(spam_data)

#  take a test message, and figure out the probability of the message being ham.print
#  sum of all prob(word|ham) then we have our guess. What threshold do we want? 
#  threshold: 51% OR 70% OR 90% ??? Depends on how strict you want it to be right?

print(pd_test_data.iloc[0])

groups = pd_training_data.groupby('labels')
hamCount = groups.get_group("ham").count().values[0]

# probability of a word being ham =  ( all messages of ham / all messages of ham + all messages of spam )

prob_ham = hamCount / len(message)
print("\nPROB OF HAM:",prob_ham,"\n")

#  MESG = pd_test_data.iloc[0]['message']
#  P( HAM | MESG ) 

#  P(HAM) * P(word[0]) * P(word[1]) * ... * P(word[n]) 
# / 
# ( P(HAM) * P(word[0]) * P(word[1]) * ... * P(word[n]) ) 
# * ( P(SPAM)  * P(wordInSpam[0]) * P(wordInSpam[1]) * ... P(wordInSpam[n])  )

test_string = pd_test_data.iloc[0]['message']
print("This is test_string: ", test_string)
# result = (prob_ham * )


from sklearn.metrics import confusion_matrix
# 1 = ham
# 0 = spam

# for e in range(len(pd_test_data)):
#     if pd_test_data.iloc[e]['labels'] == 'ham':
#         pd_test_data.iloc[e]['labels'] = 1
#     else:
#         pd_test_data.iloc[e]['labels'] = 0

# print(pd_test_data)


#  repeat this for all test messages.

# find our acccuracy based off what the message is supposed to be. (correct guesses / all test messages)