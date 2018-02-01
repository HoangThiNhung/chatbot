
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
from pyvi.pyvi import ViTokenizer
from ngram import ngrams
from underthesea import pos_tag

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random


# In[2]:


import json
with open('data/training.json') as json_data:
    intents = json.load(json_data)


# In[3]:


words = []
classes = []
documents = []
ignore_words = ['?', '!', ',', '.', 'xin_lỗi', 'và', 'ạ']

for intent in intents['data']:
    for pattern in intent['patterns']:

        w = ViTokenizer.tokenize(pattern).split(' ')
        w = [stemmer.stem(_w.lower()) for _w in w if _w not in ignore_words]
        text = ngrams(w, 4, [])
        words.extend(text)
        documents.append((text, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(words)


# In[4]:


#Create training data
training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])


# In[5]:


print(train_x[1])
print(train_y[1])


# In[6]:


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 128)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 32)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('models/model.tflearn')


# In[7]:


import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "models/training_data", "wb" ) )
