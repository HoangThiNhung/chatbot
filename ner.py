import nltk
from nltk.stem.lancaster import LancasterStemmer
from pyvi.pyvi import ViTokenizer

stemmer = LancasterStemmer()

class NER():
    def get_entity(sentence):
        entity_list = [line.rstrip('\n') for line in open('data/entity.dat')]
        sentence_words = ViTokenizer.tokenize(sentence).split(' ')

        entity = [stemmer.stem(word.lower()) for word in sentence_words if word in entity_list]
        return entity
