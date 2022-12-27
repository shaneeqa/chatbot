import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    pass



#sentence1 = "Hey there how are you doing?"

#print(tokenize(sentence1))
#execute this by: python preprocessor.py inside the virtual environment

word = "organizing"
 
#print(stem(word))

#.\chatbo_venv\Scripts\activate