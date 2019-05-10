import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable




class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()
        self.Wx = torch.randn(n_inputs, n_neurons) # 4 X 1
        self.Wy = torch.randn(n_neurons, n_neurons) # 1 X 1
        self.b = torch.zeros(1, n_neurons) # 1 X 4

    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b) # 4 X 1
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) +
                            torch.mm(X1, self.Wx) + self.b) # 4 X 1
    
        return self.Y0, self.Y1

class RNNModel(nn.Module):
    def __init__(self, vocabularys,input_dim, hidden_dim, num_layer, output_dim):
        super(RNNModel, self).__init__()

        embed = nn.Embedding(vocabularys,input_dim)
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.num_layer = num_layer
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layer, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
            
        # One time step
        out, hn = self.rnn(x)
        predict = self.fc(hn.permute(1,0,2))
        return predict


if __name__ == "__main__":
    vector_size = 50
    hidden_dim = 20
    layer_dim =2
    output_dim = 2
    embed = nn.Embedding(100,vector_size)
    rnn_tensor = torch.LongTensor(np.random.randn(3, 4)+3)
    print(rnn_tensor)
    input_dim = 50
    inputs =embed(rnn_tensor)
    RNN=RNNModel(input_dim,hidden_dim,layer_dim,output_dim)
    print(RNN.forward(inputs))
    #print(embed(rnn_tensor).size())


    



