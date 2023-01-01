import json
from preprocessor import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import NeuralNet


with open("intents.json", "r") as f:
    data = json.load(f)
    intents = data["intents"]

    # print(intents)

tags = []
all_words = []
xy = []

for intent in intents:
    tag = intent["tag"]

    tags.append(tag)

    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)

        xy.append((words, tag))  # for each of the content it will contain mapping

# print("tags: ", tags)
# print("all words: ", all_words)
# print("xy: ", xy)

ignore_words = ["?", "!", "."]

# stemming all words and removing punctuations
all_words = [stem(word) for word in all_words if word not in ignore_words]

all_words = sorted(set(all_words))  # set: select unique values out of the words

tags = sorted(set(tags))  # make sure tags doesn't have duplicatesx


# create training data

X_train = []
Y_train = []

for words, tag in xy:
    bag = bag_of_words(words, all_words)

    X_train.append(bag)

    label = tags.index(tag)

    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# storing sample data
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# training the model

# Hyperparameters
epochs = 1000  # number of trainings
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001


dataset = ChatDataSet()


train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criteria = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for bagofwords, label in train_loader:
        bagofwords = bagofwords.to(device)
        label = label.to(dtype=torch.long).to(device)

        output = model(bagofwords)

        loss = criteria(output, label)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch+1}/{epochs}, loss: {loss.item():.4f}")

print(f"final loss: loss: {loss.item():.4f}")
