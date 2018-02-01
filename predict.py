
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
from pyvi.pyvi import ViTokenizer
from ngram import ngrams
from ner import *

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

from sqlalchemy.orm import sessionmaker
from models import *

Session = sessionmaker(bind=engine)
session = Session()


# In[2]:


import pickle
data = pickle.load( open( "models/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

import json
with open('data/training.json') as json_data:
    intents = json.load(json_data)


# In[3]:


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


# In[4]:


def clean_up_sentence(sentence):
    ignore_words = ['?', '!', ',', '.', 'xin_lỗi', 'và', 'ạ']
    sentence_words = w = ViTokenizer.tokenize(sentence).split(' ')

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if word not in ignore_words]
    sentence_words = ngrams(w, 4, [])

    return sentence_words

# bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[5]:


bow('thanh toán giúp bàn mình với ạ', words)


# In[6]:


# load model
model.load('./models/model.tflearn')


# In[7]:


# data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def predict(sentence, userID='1', show_details=False):
    entity = NER.get_entity(sentence)
    results = classify(sentence)
    if results:
        while results:
            for i in intents['data']:
                if i['tag'] == results[0][0]:
                    classes = results[0][0]
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or                         (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        if classes == 'menu':
                            entity = entity[0] if len(entity) > 0 else ''
                            menu = session.query(Menu).filter(Menu.n_gram_search_text.like('%'+entity+'%')).all()
                            if len(menu) > 0:
                                print(i['responses'][0])
                                return print(menu)
                            else:
                                return print(i['responses'][1])
                        if classes == 'promotion':
                            promotion = session.query(Promotion).all()
                            if len(promotion) > 0:
                                print(i['responses'][0])
                                return print(promotion)
                            else: return print(i['responses'][1])
                        else:
                            return print(random.choice(i['responses']))

            results.pop(0)
