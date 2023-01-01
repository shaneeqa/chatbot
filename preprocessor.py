import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    
    words = [stem(wordd) for wordd in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype = np.float32)
   
    for indexx, wordd in enumerate(all_words):
        if wordd in words:
            bag[indexx] = 1
        
    return bag
    pass


#sentence1 = "Hey there how are you doing?"

#print(tokenize(sentence1))
#execute this by: python preprocessor.py inside the virtual environment

word = "organizing"
 
#print(stem(word))

#.\chatbo_venv\Scripts\activate