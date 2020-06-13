import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adamax, RMSprop

def activation_from_string(activation):
    if activation == "relu":
        return F.relu
    elif activation == "tanh":
        return F.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    else:
        return F.leaky_relu

def BuildNetwork(params):
    neurons = [params['neurons_input']]
    dropouts = []
    activations = []

    for i in range(2,5,1):
        if params['layer'+str(i)] == "yes":
            neurons += [params['neurons_layer'+str(i)]]
            activations += [activation_from_string(params['activation'+str(i)])]     
            dropouts += [params['layer'+str(i)+'_dropout'] == 'yes']

    dropouts += [False]
    neurons += [params['neurons_output']]
    dropout_prob = params['dropout_prob']

    net = HyperTuningNet(neurons, activations, dropouts, dropout_prob)
    return net

def trainNN(X_train, y_train, net, params):
    learning_rate = params['learning_rate'] 
    EPOCHS = params['epochs']
    BATCH_SIZE = params['batch_size']
    optimizer_name = params['optimizer']

    if optimizer_name == "Adam":
        optimizer = Adam(net.parameters(), lr = learning_rate)
    elif optimizer_name == "Adamax":
        optimizer = Adamax(net.parameters(), lr = learning_rate)
    else:
        optimizer = RMSprop(net.parameters(), lr = learning_rate)

    ## TRAINING
    for epoch in range(EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE):
            net.zero_grad()
            X_batch = X_train[i:i+BATCH_SIZE].float()
            y_batch = y_train[i:i+BATCH_SIZE]

            y_preds = net(X_batch.view(-1, X_train.shape[1]))
            y_preds.float()
            loss = F.nll_loss(y_preds, y_batch)
            loss.backward()
            optimizer.step()
            
    return net

def report_scoreNN(net, X_test, y_test, params, new_dict):
    correct = 0
    total = 0
    BATCH_SIZE = params['batch_size']

    with torch.no_grad():
        for i in range(0, len(y_test), BATCH_SIZE):
            X_batch = X_test[i:i+BATCH_SIZE].float()
            y_batch = y_test[i:i+BATCH_SIZE]
            net.eval()
            y_preds = torch.argmax(net(X_batch.view(-1, X_test.shape[1])), dim=1)
            for j, y_pred in enumerate(y_preds):
                if y_batch[j].item() in new_dict[y_pred.item()]:
                    correct += 1
                total += 1
    return correct/total

class HyperTuningNet(nn.Module): #This is a network whose architecture is tuned using hyperopt
    
    def __init__(self, neurons, activations, dropouts, dropout_prob):
        super().__init__()
        
        layers = []
        self.dropouts = []
        for i in range(0, len(neurons)-1,1):
            if dropouts[i]:
                self.dropouts += [nn.Dropout(p=dropout_prob)]
            else:
                self.dropouts += [nn.Dropout(p=0)]
            layers += [nn.Linear(neurons[i], neurons[i+1])]

        self.layers = nn.ModuleList(layers)
        self.activations = activations

    def forward(self, x):
        for i in range(0, len(self.layers)-1, 1):
            layer = self.layers[i]
            activation =  self.activations[i]
            dropout = self.dropouts[i]
            x = dropout(activation(layer(x)))
        
        layer = self.layers[-1]
        x = F.softmax(layer(x), dim=1)
        return x
    
    def predict(self, x): #can only predict one test point at a time
        
        x = torch.Tensor(x)
        with torch.no_grad():
            self.eval()
            y_pred = torch.argmax(self(x.view(-1, x.shape[1])), dim=1)
        return y_pred

