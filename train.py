import json
from preprocessor import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
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

tags = sorted(set(tags)) #make sure tags doesn't have duplicatesx


#create training data

X_train = []
Y_train = []

for words, tag in xy:
    bag = bag_of_words(words)

    X_train.append(bag)

    label = tags.index[tag]

    Y_train.append(label)

#storing sample data
class ChatDataSet(Dataset)
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train 

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.n_samples
        
#Hyperparameters
batch_size = 8

dataset = ChatDataSet()

dataset(0)
train_loader = DataLoader(dataset = , batch_size = batch_size, shuffle = True, num_workers = 0)
#training the model