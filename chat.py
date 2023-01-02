import torch
from model import NeuralNet
from preprocessor import tokenize, bag_of_words

FILE = "model_data.pth"

model_data = torch.load(FILE)

# print(model_data)

model_state = model_data['model_state']

input_size = model_data['input_size']

hidden_size = model_data['hidden_size']

output_size = model_data['output_size']

all_words = model_data['all_words']

tags = model_data['tags']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval() # activate the model


def response(userInput):
    sentence = tokenize(userInput)
    
    X = bag_of_words(sentence, all_words)
    print(X)

if __name__ == "__main__":
    print("Let's start chatting! type (quit) to exit the chat")

    while True:
        userInput = input("User: ")

        if(userInput) == "quit":
            break
        
        print(response(userInput))