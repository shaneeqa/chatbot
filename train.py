import json
from preprocessor import tokenize, stem, bag_of_words
import numpy

with open('intents.json', 'r') as f:
    data = json.load(f)
    intents = data['intents']

    # print(intents)

tags =[]
all_words = []
xy = []

for intent in intents:
    tag = intent['tag']

    tags.append(tag)

    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)

        xy.append((words, tag)) #for each of the content it will contain mapping

# print("tags: ", tags)
# print("all words: ", all_words)
# print("xy: ", xy)

ignore_words = ['?', '!', '.']

#stemming all words and removing punctuations
all_words = [stem(word) for word in all_words if word not in ignore_words]

all_words = sorted(set(all_words)) #set: select unique values out of the words

tags = set(tags) #make sure tags doesn't have duplicatesx
