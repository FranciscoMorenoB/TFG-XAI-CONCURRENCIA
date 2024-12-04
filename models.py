from keras.models import *
from keras.layers import *
from keras.callbacks import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from keras.models import *
from keras.layers import *
from keras.callbacks import *


class CNN_Model(nn.Module):
   
    def __init__(self, data, filters_l1=32, filters_l2=64, maxpool_1=4, maxpool_2=-1, kernel_initializer='default'):
        super(CNN_Model, self).__init__()
        
        kernel_size = data.num_one_hot_encodings * 4
        padding_size = int(kernel_size / 2)
        self.cnnf1_1 = nn.Conv1d(1, filters_l1, kernel_size=kernel_size,
                              stride=data.num_one_hot_encodings, padding=padding_size)
        self.cnnf2_1 = nn.Conv1d(1, filters_l1, kernel_size=kernel_size,
                              stride=data.num_one_hot_encodings, padding=padding_size)
        self.cnnf3_1 = nn.Conv1d(1, filters_l1, kernel_size=kernel_size,
                              stride=data.num_one_hot_encodings, padding=padding_size)
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size = self.cnnf1_1(random_sample).shape
#         print(dense_input_size)
        
        self.cnnf1_2 = nn.Conv1d(filters_l1, filters_l1, kernel_size=2, padding=1)
        self.cnnf2_2 = nn.Conv1d(filters_l1, filters_l1, kernel_size=2, padding=1)
        self.cnnf3_2 = nn.Conv1d(filters_l1, filters_l1, kernel_size=2, padding=1)
        
        
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size =self.cnnf1_2(self.cnnf1_1(random_sample)).shape
#         print(dense_input_size)
        
        self.maxf1 = nn.MaxPool1d(maxpool_1)
        self.maxf2 = nn.MaxPool1d(maxpool_1)
        self.maxf3 = nn.MaxPool1d(maxpool_1)
        
        
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size = self.maxf1(self.cnnf1_2(self.cnnf1_1(random_sample))).shape
#         print(dense_input_size)        
        
        self.cnnf1_3 = nn.Conv1d(filters_l1, filters_l2, kernel_size=2, padding=1)
        self.cnnf2_3 = nn.Conv1d(filters_l1, filters_l2, kernel_size=2, padding=1)
        self.cnnf3_3 = nn.Conv1d(filters_l1, filters_l2, kernel_size=2, padding=1)
        
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size = self.cnnf1_3(self.maxf1(self.cnnf1_2(self.cnnf1_1(random_sample)))).shape
#         print(dense_input_size)
        
        
        self.cnnf1_4 = nn.Conv1d(filters_l2, filters_l2, kernel_size=2, padding=1)
        self.cnnf2_4 = nn.Conv1d(filters_l2, filters_l2, kernel_size=2, padding=1)
        self.cnnf3_4 = nn.Conv1d(filters_l2, filters_l2, kernel_size=2, padding=1)       
        
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size = self.cnnf1_4(self.cnnf1_3(self.maxf1(self.cnnf1_2(self.cnnf1_1(random_sample))))).shape
#         print(dense_input_size)
    
        if maxpool_2 == -1:
            maxpool_2 = dense_input_size[-1]
        
        self.maxf1_2 = nn.MaxPool1d(maxpool_2)
        self.maxf2_2 = nn.MaxPool1d(maxpool_2)
        self.maxf3_2 = nn.MaxPool1d(maxpool_2)

#         self.maxf1_2 = nn.MaxPool1d(2)
#         self.maxf2_2 = nn.MaxPool1d(2)
#         self.maxf3_2 = nn.MaxPool1d(2)
        
        random_sample = torch.randn(1, 1, data.num_one_hot_encodings * data.layer_size)
        dense_input_size = self.maxf1_2(self.cnnf1_4(self.cnnf1_3(self.maxf1(self.cnnf1_2(self.cnnf1_1(random_sample))))))
#         print(dense_input_size.shape)

        self.densef1 = nn.Linear(dense_input_size.shape[1] * dense_input_size.shape[2], 8)
        self.densef2 = nn.Linear(dense_input_size.shape[1] * dense_input_size.shape[2], 8)
        self.densef3 = nn.Linear(dense_input_size.shape[1] * dense_input_size.shape[2], 8)
        
        self.dense = nn.Linear(8 * 3, 1, )
        
        if kernel_initializer == 'zeros':
            for name, param in self.named_parameters(): 
                torch.nn.init.zeros_(param)
        elif kernel_initializer == 'keras':
            def initialize_keras(model):
                if type(model) in [nn.Linear, nn.Conv1d]:
                    nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(model.bias)
            self.apply(initialize_keras)
            
    def forward(self, x_f1, x_f2, x_f3):
        
        x_f1 = x_f1.view(x_f1.shape[0], 1, -1)
        x_f2 = x_f2.view(x_f2.shape[0], 1, -1)
        x_f3 = x_f3.view(x_f3.shape[0], 1, -1)

        x_f1 = F.relu(self.cnnf1_1(x_f1))
        x_f2 = F.relu(self.cnnf2_1(x_f2))
        x_f3 = F.relu(self.cnnf3_1(x_f3))
        
        x_f1 = F.relu(self.cnnf1_2(x_f1))
        x_f2 = F.relu(self.cnnf2_2(x_f2))
        x_f3 = F.relu(self.cnnf3_2(x_f3))
        
        x_f1 = self.maxf1(x_f1)
        x_f2 = self.maxf2(x_f2)
        x_f3 = self.maxf3(x_f3)
        
        x_f1 = F.relu(self.cnnf1_3(x_f1))
        x_f2 = F.relu(self.cnnf2_3(x_f2))
        x_f3 = F.relu(self.cnnf3_3(x_f3))
        
        x_f1 = F.relu(self.cnnf1_4(x_f1))
        x_f2 = F.relu(self.cnnf2_4(x_f2))
        x_f3 = F.relu(self.cnnf3_4(x_f3))
        
        x_f1 = self.maxf1_2(x_f1)
        x_f2 = self.maxf2_2(x_f2)
        x_f3 = self.maxf3_2(x_f3)
        
        x_f1 = x_f1.view(x_f1.shape[0], -1)
        x_f2 = x_f2.view(x_f2.shape[0], -1)
        x_f3 = x_f3.view(x_f3.shape[0], -1)
        
        x_f1 = self.densef1(x_f1)
        x_f2 = self.densef2(x_f2)
        x_f3 = self.densef3(x_f3)
        
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        
        x = self.dense(x)
        return x

def get_lstm_keras(data, neurons_l1=32, neurons_l2=16, neurons_l3=8, 
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', optimizer='adam'):

    input_f1 = Input(shape=(int((data.f2_start - data.f1_start) / data.num_one_hot_encodings), data.num_one_hot_encodings))
    input_f2 = Input(shape=(int((data.f3_start - data.f2_start) / data.num_one_hot_encodings), data.num_one_hot_encodings))
    input_f3 = Input(shape=(int((data.f3_end - data.f3_start) / data.num_one_hot_encodings), data.num_one_hot_encodings))

    f1_l1 = LSTM(neurons_l1, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer)(input_f1)
    f2_l1 = LSTM(neurons_l1, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer)(input_f2)
    f3_l1 = LSTM(neurons_l1, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer)(input_f3)

    f1_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f1_l1)
    f2_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f2_l1)
    f3_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f3_l1)
    
    f1_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f1_l2)
    f2_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f2_l2)
    f3_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f3_l2)

    concat = concatenate([f1_l3, f2_l3, f3_l3])
    dense = Dense(8, activation='relu', kernel_initializer=kernel_initializer)(concat)
    out = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)(dense)

    model = Model(inputs=[input_f1, input_f2, input_f3], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model



class LSTM_Model(nn.Module):
   
    def __init__(self, data, neurons_l1=32, neurons_l2=16, neurons_l3=8, kernel_initializer='default'):
        super(LSTM_Model, self).__init__()
        
        feature_size = data.num_one_hot_encodings

        self.lstm1 = nn.LSTM(feature_size, neurons_l1, batch_first=True)
        self.lstm2 = nn.LSTM(feature_size, neurons_l1, batch_first=True)
        self.lstm3 = nn.LSTM(feature_size, neurons_l1, batch_first=True)

        self.densef1_1 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_1 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_1 = nn.Linear(neurons_l1, neurons_l2)
        
        self.densef1_2 = nn.Linear(neurons_l2, neurons_l3)
        self.densef2_2 = nn.Linear(neurons_l2, neurons_l3)
        self.densef3_2 = nn.Linear(neurons_l2, neurons_l3)
        
        self.dense_1 = nn.Linear(neurons_l3 * 3, 8)
        self.dense_2 = nn.Linear(8, 1)
        
        if kernel_initializer == 'zeros':
            for name, param in self.named_parameters(): 
                torch.nn.init.zeros_(param)
        elif kernel_initializer == 'keras':
            def initialize_keras(model):
                if type(model) in [nn.Linear]:
                    nn.init.xavier_uniform_(model.weight)
                    nn.init.zeros_(model.bias)
                elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                    nn.init.orthogonal_(model.weight_hh_l0)
                    nn.init.xavier_uniform_(model.weight_ih_l0)
                    nn.init.zeros_(model.bias_hh_l0)
                    nn.init.zeros_(model.bias_ih_l0)
            self.apply(initialize_keras)

    def forward(self, x_f1, x_f2, x_f3):
        lstm_outf1, lstm_hidden1 = self.lstm1(x_f1)
        lstm_outf2, lstm_hidden2 = self.lstm2(x_f2)
        lstm_outf3, lstm_hidden3 = self.lstm3(x_f3)

        x_f1 = F.relu(self.densef1_1(torch.tanh(lstm_hidden1[0].view(lstm_hidden1[0].shape[1], -1))))
        x_f2 = F.relu(self.densef2_1(torch.tanh(lstm_hidden2[0].view(lstm_hidden2[0].shape[1], -1))))
        x_f3 = F.relu(self.densef3_1(torch.tanh(lstm_hidden3[0].view(lstm_hidden3[0].shape[1], -1))))

        x_f1 = F.relu(self.densef1_2(x_f1))
        x_f2 = F.relu(self.densef2_2(x_f2))
        x_f3 = F.relu(self.densef3_2(x_f3))
                
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x



def get_deepsets_keras(data, optimizer='adam', neurons_l1=128, neurons_l2=32, neurons_l3=8):
    
    input_f1 = Input(shape=(data.f2_start - data.f1_start,))
    input_f2 = Input(shape=(data.f3_start - data.f2_start,))
    input_f3 = Input(shape=(data.f3_end - data.f3_start,))

    f1_l1 = Dense(neurons_l1, activation='relu')(input_f1)
    f2_l1 = Dense(neurons_l1, activation='relu')(input_f2)
    f3_l1 = Dense(neurons_l1, activation='relu')(input_f3)

    f1_l2 = Dense(neurons_l2, activation='relu')(f1_l1)
    f2_l2 = Dense(neurons_l2, activation='relu')(f2_l1)
    f3_l2 = Dense(neurons_l2, activation='relu')(f3_l1)
    
    f1_l3 = Dense(neurons_l3, activation='relu')(f1_l2)
    f2_l3 = Dense(neurons_l3, activation='relu')(f2_l2)
    f3_l3 = Dense(neurons_l3, activation='relu')(f3_l2)
        
    added = Add()([f1_l3, f2_l3, f3_l3])    
    
    concat_1 = concatenate([f1_l3, f2_l3, f3_l3])
    dense = Dense(8, activation='relu')(concat_1)
    
    concat_2 = concatenate([added, dense])
    out = Dense(1, activation='sigmoid')(concat_2)
    model = Model(inputs=[input_f1, input_f2, input_f3], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

class DEEPSET_Model(nn.Module):
   
    def __init__(self, data, neurons_l1=128, neurons_l2=32, neurons_l3=8, kernel_initializer='default'):
        super(DEEPSET_Model, self).__init__()

        self.densef1_1 = nn.Linear(data.f2_start - data.f1_start, neurons_l1)
        self.densef2_1 = nn.Linear(data.f3_start - data.f2_start, neurons_l1)
        self.densef3_1 = nn.Linear(data.f3_end - data.f3_start, neurons_l1)
        
        self.densef1_2 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_2 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_2 = nn.Linear(neurons_l1, neurons_l2)
        
        self.densef1_3 = nn.Linear(neurons_l2, neurons_l3)
        self.densef2_3 = nn.Linear(neurons_l2, neurons_l3)
        self.densef3_3 = nn.Linear(neurons_l2, neurons_l3)
        
        self.dense_1 = nn.Linear(neurons_l3 * 3, 8)
        self.dense_2 = nn.Linear(8 * 2, 1)
        
        if kernel_initializer == 'zeros':
            for name, param in self.named_parameters(): 
                torch.nn.init.zeros_(param)
        elif kernel_initializer == 'keras':
            def initialize_keras(model):
                if type(model) in [nn.Linear]:
                    nn.init.xavier_uniform_(model.weight)
                    nn.init.zeros_(model.bias)
                elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                    nn.init.orthogonal_(model.weight_hh_l0)
                    nn.init.xavier_uniform_(model.weight_ih_l0)
                    nn.init.zeros_(model.bias_hh_l0)
                    nn.init.zeros_(model.bias_ih_l0)
            self.apply(initialize_keras)

    def forward(self, x_f1, x_f2, x_f3):
        
        x_f1 = x_f1.view(x_f1.shape[0], -1)
        x_f2 = x_f2.view(x_f2.shape[0], -1)
        x_f3 = x_f3.view(x_f3.shape[0], -1)
        
        x_f1 = F.relu(self.densef1_1(x_f1))
        x_f2 = F.relu(self.densef2_1(x_f2))
        x_f3 = F.relu(self.densef3_1(x_f3))

        x_f1 = F.relu(self.densef1_2(x_f1))
        x_f2 = F.relu(self.densef2_2(x_f2))
        x_f3 = F.relu(self.densef3_2(x_f3))
       
        x_f1 = F.relu(self.densef1_3(x_f1))
        x_f2 = F.relu(self.densef2_3(x_f2))
        x_f3 = F.relu(self.densef3_3(x_f3))
               
        x_deepsets = x_f1 + x_f2 + x_f3
        x_concat = torch.cat((x_f1, x_f2, x_f3), 1)
        x_dense = F.relu(self.dense_1(x_concat))
        
        x = torch.cat((x_deepsets, x_dense), 1)
        x = self.dense_2(x)
        return x
    
class DEEPSETV2_Model(nn.Module):
   
    def __init__(self, data, neurons_l1=32, neurons_l2=8, kernel_initializer='default'):
        super(DEEPSETV2_Model, self).__init__()

        self.densef1_1 = nn.Linear(data.f2_start - data.f1_start, 1)
        self.densef2_1 = nn.Linear(data.f3_start - data.f2_start, 1)
        self.densef3_1 = nn.Linear(data.f3_end - data.f3_start, 1)
        
        self.densef1_2 = nn.Linear(1, neurons_l1)
        self.densef2_2 = nn.Linear(1, neurons_l1)
        self.densef3_2 = nn.Linear(1, neurons_l1)
        
        self.densef1_3 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_3 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_3 = nn.Linear(neurons_l1, neurons_l2)
        
        self.dense_1 = nn.Linear(neurons_l2 * 3, 1)
        
        if kernel_initializer == 'zeros':
            for name, param in self.named_parameters(): 
                torch.nn.init.zeros_(param)
        elif kernel_initializer == 'keras':
            def initialize_keras(model):
                if type(model) in [nn.Linear]:
                    nn.init.xavier_uniform_(model.weight)
                    nn.init.zeros_(model.bias)
                elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                    nn.init.orthogonal_(model.weight_hh_l0)
                    nn.init.xavier_uniform_(model.weight_ih_l0)
                    nn.init.zeros_(model.bias_hh_l0)
                    nn.init.zeros_(model.bias_ih_l0)
            self.apply(initialize_keras)

    def forward(self, x_f1, x_f2, x_f3):
        
        x_f1 = x_f1.view(x_f1.shape[0], -1)
        x_f2 = x_f2.view(x_f2.shape[0], -1)
        x_f3 = x_f3.view(x_f3.shape[0], -1)
        
        x_f1 = F.relu(self.densef1_1(x_f1))
        x_f2 = F.relu(self.densef2_1(x_f2))
        x_f3 = F.relu(self.densef3_1(x_f3))

        x_f1 = F.relu(self.densef1_2(x_f1))
        x_f2 = F.relu(self.densef2_2(x_f2))
        x_f3 = F.relu(self.densef3_2(x_f3))
       
        x_f1 = F.relu(self.densef1_3(x_f1))
        x_f2 = F.relu(self.densef2_3(x_f2))
        x_f3 = F.relu(self.densef3_3(x_f3))
               
        x_concat = torch.cat((x_f1, x_f2, x_f3), 1)

        x = F.relu(self.dense_1(x_concat))
        
        return x

def get_feedforward_keras(data, optimizer='adam', neurons_l1=128, neurons_l2=32, neurons_l3=8, kernel_initializer='glorot_uniform'):
    
    input_f1 = Input(shape=(data.f2_start - data.f1_start,))
    input_f2 = Input(shape=(data.f3_start - data.f2_start,))
    input_f3 = Input(shape=(data.f3_end - data.f3_start,))

    f1_l1 = Dense(neurons_l1, activation='relu', kernel_initializer=kernel_initializer)(input_f1)
    f2_l1 = Dense(neurons_l1, activation='relu', kernel_initializer=kernel_initializer)(input_f2)
    f3_l1 = Dense(neurons_l1, activation='relu', kernel_initializer=kernel_initializer)(input_f3)

    f1_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f1_l1)
    f2_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f2_l1)
    f3_l2 = Dense(neurons_l2, activation='relu', kernel_initializer=kernel_initializer)(f3_l1)
    
    f1_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f1_l2)
    f2_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f2_l2)
    f3_l3 = Dense(neurons_l3, activation='relu', kernel_initializer=kernel_initializer)(f3_l2)
    
   
    concat = concatenate([f1_l3, f2_l3, f3_l3])
    dense = Dense(8, activation='relu', kernel_initializer=kernel_initializer)(concat)
    out = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)(dense)

    model = Model(inputs=[input_f1, input_f2, input_f3], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


class FEEDFORWARD_Model(nn.Module):
   
    def __init__(self, data, neurons_l1=128, neurons_l2=32, neurons_l3=8, kernel_initializer='default'):
        super(FEEDFORWARD_Model, self).__init__()
                
        self.densef1_1 = nn.Linear(data.f2_start - data.f1_start, neurons_l1)
        self.densef2_1 = nn.Linear(data.f3_start - data.f2_start, neurons_l1)
        self.densef3_1 = nn.Linear(data.f3_end - data.f3_start, neurons_l1)
        
        self.densef1_2 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_2 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_2 = nn.Linear(neurons_l1, neurons_l2)
        
        self.densef1_3 = nn.Linear(neurons_l2, neurons_l3)
        self.densef2_3 = nn.Linear(neurons_l2, neurons_l3)
        self.densef3_3 = nn.Linear(neurons_l2, neurons_l3)
        
        self.dense_1 = nn.Linear(neurons_l3 * 3, 8)
        self.dense_2 = nn.Linear(8, 1)
        
        if kernel_initializer == 'zeros':
            for name, param in self.named_parameters(): 
                torch.nn.init.zeros_(param)
        elif kernel_initializer == 'keras':
            def initialize_keras(model):
                if type(model) in [nn.Linear]:
                    nn.init.xavier_uniform_(model.weight)
                    nn.init.zeros_(model.bias)
                elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                    nn.init.orthogonal_(model.weight_hh_l0)
                    nn.init.xavier_uniform_(model.weight_ih_l0)
                    nn.init.zeros_(model.bias_hh_l0)
                    nn.init.zeros_(model.bias_ih_l0)
            self.apply(initialize_keras)

    def forward(self, x_f1, x_f2, x_f3):
        
        x_f1 = x_f1.view(x_f1.shape[0], -1)
        x_f2 = x_f2.view(x_f2.shape[0], -1)
        x_f3 = x_f3.view(x_f3.shape[0], -1)
        
        x_f1 = F.relu(self.densef1_1(x_f1))
        x_f2 = F.relu(self.densef2_1(x_f2))
        x_f3 = F.relu(self.densef3_1(x_f3))

        x_f1 = F.relu(self.densef1_2(x_f1))
        x_f2 = F.relu(self.densef2_2(x_f2))
        x_f3 = F.relu(self.densef3_2(x_f3))
       
        x_f1 = F.relu(self.densef1_3(x_f1))
        x_f2 = F.relu(self.densef2_3(x_f2))
        x_f3 = F.relu(self.densef3_3(x_f3))
                
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x
    
    
    
def train_epoch(model, train_X, train_Y, criterion, optimizer, epoch, batch_size, verbose=False): #batch_size: El tamaño de los lotes en los que se divide el conjunto de datos para el entrenamiento.

    model.train() #indica que el modelo está en modo entren, ya que algunas capas como Dropout o BatchNorm se comportan de manera diferente durante el entrenamiento y la evaluación. Dropout desactiva ciertas neuronas al azar durante el entrenamiento
    total_loss = 0 #perdida total
    correct = 0# num de predicciones correctas
    num_samples = train_X[0].shape[0] # numero de f1 = num de muestras 


    #Se recorre el conjunto de entrenamiento en bloques de tamaño batch_size. 
    # El número de lotes es calculado dividiendo el número total de muestras entre el tamaño del lote, con un ajuste para cubrir el caso en que el número de muestras no sea un múltiplo exacto de batch_size.
    
    for i in range(int((num_samples - 1) / batch_size) + 1):
        # Reset gradient data to 0
        
        #los gradientes sirven para ajustar los parámetros del modelo (como los pesos y los sesgos) de manera que la función de pérdida se minimice y el modelo aprenda a realizar predicciones más precisas.
        #Los gradientes nos indican cuánto debe cambiar cada peso para reducir el error (función de pérdida).
        #La magnitud del gradiente indica qué tan grande debe ser el cambio.
        #La dirección del gradiente indica hacia qué dirección deben moverse los pesos.

        #El gradiente de la función de pérdida LL con respecto a un parámetro ww se denota como: (derivadas)  #∂L / ∂w . Si el gradiente es grande, significa que el modelo está muy sensible a ese peso y que un pequeño cambio puede tener un gran efecto en el resultado

        #Retropropagación es el algoritmo que calcula estos gradientes en redes neuronales.


        optimizer.zero_grad() #los gradientes (param) deben ser restablecidos a cero porque PyTorch acumula gradientes por defecto.

        #optimizer.zero_grad(): Pone los gradientes de todos los parámetros del modelo a cero. Es necesario porque, de lo contrario, los gradientes de diferentes pasos se acumularían.
        #loss.backward(): (antes se calcula la funcion de perdida) Calcula los gradientes de la función de pérdida con respecto a los parámetros del modelo (pesos y sesgo).
        #optimizer.step(): Actualiza los pesos del modelo utilizando los gradientes calculados por loss.backward(). Los parámetros del modelo se actualizan utilizando el algoritmo de optimización ADAM
        

        #seleccionar el índice del lote actuala segura que no se intente acceder a índices fuera de rango.
        if (i+1) * batch_size > num_samples:
            batch_slice = slice(i * batch_size, num_samples)
        else:
            batch_slice = slice(i * batch_size, (i+1)*batch_size)
            
  
        # 1) Get the prediction for batch
        output = model(train_X[0][batch_slice], train_X[1][batch_slice], train_X[2][batch_slice]).view(-1) #view(-1) significa que el tensor resultante será aplanado en una sola dimensión (un vector unidimensional)

        target = train_Y[i * batch_size:(i+1)*batch_size]

        # 2) calcula la perdidia
        loss = criterion(output, target)

        # 3) retropropagación, calculo de los gradientes
        loss.backward()

        # 4) Update los parametros
        optimizer.step()

        #calcular la precicisón del lote
        preds_binary = torch.sigmoid(output) >= 0.5
        target_binary = target >= 0.5

        total_loss += loss.detach()

        for j in range(preds_binary.shape[0]):
            if preds_binary[j] == target_binary[j]:
                correct += 1
             
    if verbose:    
        print('Train Epoch: {} LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
            epoch, total_loss / num_samples * batch_size, correct, num_samples, 100. * correct / num_samples))
        
    return (total_loss / num_samples * batch_size).item(), (correct / num_samples)
    
def eval_epoch(model, X, Y, criterion, name, verbose=False):
    model.eval()
    correct = 0
    num_samples = Y.shape[0]
    
    output = model(X[0], X[1], X[2]).view(-1)
    
    loss = criterion(output, Y)
    
    preds_binary = torch.sigmoid(output) >= 0.5
    target_binary = Y >= 0.5
    
    for j in range(preds_binary.shape[0]):
        if preds_binary[j] == target_binary[j]:
            correct += 1
            
    if verbose:
        print('{} set: LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)\n'.format(
            name, loss, correct, num_samples, 100. * correct / num_samples))
    return loss.item(), correct / num_samples
