import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #hidden layer size
        super(NeuralNet, self)
        self.l1 = nn.Linear(input_size, hidden_size) #level 1 regression
        self.l2 = nn.Linear(hidden_size, hidden_size) #level 2 regression
        self.l3 = nn.Linear(hidden_size,output_size)
        #above 3 are objects
        self.relu = nn.ReLu #for logistic regression

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out) #need to be sent level 2

        #level regression
        out = self.l2(out)
        out = self.relu(out)

        out = self.l3(out)
        