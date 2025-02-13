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
        
        #-------------------------------------------------------------CAMBIO
        self.dense = nn.Linear(8 * 3, 4) #Nosotros hacemos una clasificacion multiclase (4 clases)
        
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
              # Transformar las entradas a la forma adecuada   
        x_f1 = x_f1.view(x_f1.shape[0], 1, -1)
        x_f2 = x_f2.view(x_f2.shape[0], 1, -1)
        x_f3 = x_f3.view(x_f3.shape[0], 1, -1)

        # Propagación hacia adelante por las capas convolucionales y max pooling 
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
        
        # Aplanar las salidas para pasarlas a las capas densas 
        x_f1 = x_f1.view(x_f1.shape[0], -1)
        x_f2 = x_f2.view(x_f2.shape[0], -1)
        x_f3 = x_f3.view(x_f3.shape[0], -1)
        
        x_f1 = self.densef1(x_f1)
        x_f2 = self.densef2(x_f2)
        x_f3 = self.densef3(x_f3)
        

        # Concatenar las salidas de las tres ramas (f1, f2, f3) 
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        
         # Capa final densa para obtener las predicciones de clase 
        x = self.dense(x)

        # Aplicar softmax para obtener probabilidades de clase
        # x = F.softmax(x, dim=1)  # Softmax sobre las clases

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
    out = Dense(4, activation='softmax', kernel_initializer=kernel_initializer)(dense)

    model = Model(inputs=[input_f1, input_f2, input_f3], outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model



class LSTM_Model(nn.Module):
   
    def __init__(self, data, neurons_l1=32, neurons_l2=16, neurons_l3=8, kernel_initializer='default'):
        super(LSTM_Model, self).__init__()
        
        print("HOOOOOLA")


        feature_size = data.num_one_hot_encodings

        print("Número de características de entrada por paso de tiempo", feature_size) # si la entrada tiene la forma (26, 6, 8) el feature_size debería ser 8

        self.lstm1 = nn.LSTM(feature_size, neurons_l1, batch_first=True)
        self.lstm2 = nn.LSTM(feature_size, neurons_l1, batch_first=True)
        self.lstm3 = nn.LSTM(feature_size, neurons_l1, batch_first=True)


        self.densef1_1 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_1 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_1 = nn.Linear(neurons_l1, neurons_l2)
        
        self.densef1_2 = nn.Linear(neurons_l2, neurons_l3)
        self.densef2_2 = nn.Linear(neurons_l2, neurons_l3)
        self.densef3_2 = nn.Linear(neurons_l2, neurons_l3)
        
        self.dense_1 = nn.Linear(neurons_l3 * 3, 8) #La capa luego reduce este tamaño a 8.
        #-------------------------------------------------------------CAMBIO
        
        #en lugar de reducir el tamaño a 8, lo reduce a 4.
        #lo que significa que espera una entrada con un tamaño de características de neurons_l3 * 3 ( 8 * 3 = 24) y produce una salida de tamaño 4
        self.dense_2 = nn.Linear(neurons_l3, 4) # neurons_l3 (8) *3 porque estamos concatenando 3 salidas de tamño 8. Nosotros hacemos una clasificacion multiclase (4 clases)
        #self.dense_2 = nn.Linear(8, 4)
        
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
        #out no se usa ??
        lstm_outf1, lstm_hidden1 = self.lstm1(x_f1)
        lstm_outf2, lstm_hidden2 = self.lstm2(x_f2)
        lstm_outf3, lstm_hidden3 = self.lstm3(x_f3)

        print("------------------------------------------------------")

        print("lstm_outf1 shape:", lstm_outf1.shape)
        print("lstm_outf2 shape:", lstm_outf2.shape)
        print("lstm_outf3 shape:", lstm_outf3.shape)
        print("------------------------------------------------------")


        x_f1 = F.relu(self.densef1_1(torch.tanh(lstm_hidden1[0].view(lstm_hidden1[0].shape[1], -1))))
        x_f2 = F.relu(self.densef2_1(torch.tanh(lstm_hidden2[0].view(lstm_hidden2[0].shape[1], -1))))
        x_f3 = F.relu(self.densef3_1(torch.tanh(lstm_hidden3[0].view(lstm_hidden3[0].shape[1], -1))))

        print("x_f1 shape after densef1_1:", x_f1.shape)
        print("x_f2 shape after densef2_1:", x_f2.shape)
        print("x_f3 shape after densef3_1:", x_f3.shape)
        print("------------------------------------------------------")


        x_f1 = F.relu(self.densef1_2(x_f1))
        x_f2 = F.relu(self.densef2_2(x_f2))
        x_f3 = F.relu(self.densef3_2(x_f3))
                
        print("x_f1 shape after densef1_2:", x_f1.shape)
        print("x_f2 shape after densef2_2:", x_f2.shape)
        print("x_f3 shape after densef3_2:", x_f3.shape)
        print("------------------------------------------------------")


        # Concatenar las salidas de las tres LSTMs
        x = torch.cat((x_f1, x_f2, x_f3), 1)

        #esperado x_f1 shape: torch.Size([batch_size, 8]) sí
        print("x_f1 shape:", x_f1.shape)
        print("x_f2 shape:", x_f2.shape)
        print("x_f3 shape:", x_f3.shape)
        print("------------------------------------------------------")
        print("x shape after concatenation:", x.shape)


        x = F.relu(self.dense_1(x))

        print("x shape after dense_1:", x.shape)
        print("------------------------------------------------------")

        y = self.dense_2(x)
        print("x shape after dense_2:", y.shape)

        print("-------------ee-----------------------------------------")

        return y



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
    out = Dense(1, activation='softmax')(concat_2)
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

         # --------------------------------------------POSIBLE CAMBIO ----------self.dense_2 = nn.Linear(8 * 3, 4
        self.dense_2 = nn.Linear(8 * 2, 4)
        
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

        #-------------------------------------------------------CAMBIO 
        self.densef1_1 = nn.Linear(data.f2_start - data.f1_start, 1)
        self.densef2_1 = nn.Linear(data.f3_start - data.f2_start, 1)
        self.densef3_1 = nn.Linear(data.f3_end - data.f3_start, 1)
        
        self.densef1_2 = nn.Linear(1, neurons_l1)
        self.densef2_2 = nn.Linear(1, neurons_l1)
        self.densef3_2 = nn.Linear(1, neurons_l1)
        
        self.densef1_3 = nn.Linear(neurons_l1, neurons_l2)
        self.densef2_3 = nn.Linear(neurons_l1, neurons_l2)
        self.densef3_3 = nn.Linear(neurons_l1, neurons_l2)
        
        #-------------------------------------------------------CAMBIO 4
        self.dense_1 = nn.Linear(neurons_l2 * 3, 4)
        
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
    out = Dense(1, activation='softmax', kernel_initializer=kernel_initializer)(dense)

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
        #-------------------------------------------------------CAMBIO
        self.dense_2 = nn.Linear(8, 4)
        
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
    num_samples = train_X[0].shape[0] # numero de muestras = num filas
    #x_train[0][2] #[x][y] accedes fx de la muestra y    max [2][25]
    #x_train[0].shape #devuelve un par que indica (num de muestras, numero de funciones en total = 3*numMuestras)

    #Se recorre el conjunto de entrenamiento en bloques de tamaño batch_size. 
    # El número de lotes es calculado dividiendo el número total de muestras entre el tamaño del lote, con un ajuste para cubrir el caso en que el número de muestras no sea un múltiplo exacto de batch_size.
    


        # Convertir train_Y a tensor de PyTorch si no lo es
    #if not isinstance(train_Y, torch.Tensor):
    #    train_Y = torch.tensor(train_Y, dtype=torch.long)  # Asegurarse de que sea un tensor de enteros

    cont = 0 #para depurar

    for i in range(int((num_samples - 1) / batch_size) + 1):
        cont = cont+1
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
        

        #seleccionar el índice del lote actual a segura que no se intente acceder a índices fuera de rango.
        if (i+1) * batch_size > num_samples:
            batch_slice = slice(i * batch_size, num_samples)
        else:
            batch_slice = slice(i * batch_size, (i+1)*batch_size)

        print(f"Lote {i+1}: Tamaño del lote: {train_X[0][batch_slice].shape[0]}")

        # 1) Get the prediction for batch
        #le pasas una muestra basicamente no? en pplan f1, f2, f3 de la misma
        #le pasas 3 muestras de las que saca multiples entradas ??? cada op es una entrada
        output = model(train_X[0][batch_slice], train_X[1][batch_slice], train_X[2][batch_slice]) #.view(-1) #view(-1) significa que el tensor resultante será aplanado en una sola dimensión (un vector unidimensional)
        
        if (cont == 1):
            print("Forma de la salida del modelo:", output.shape) # salida (batch_size, num_classes): torch.Size([16, 4]) ignifica que la forma del tensor de salida del modelo tiene dos dimensiones: 
        #6: El número de muestras o ejemplos en el lote. Esto es el tamaño del batch de datos que estás pasando a través del modelo. Cada fila del tensor corresponde a una muestra o ejemplo diferente.
        #4: El número de clases posibles para cada muestra. Esto corresponde al número de clases en un problema de clasificación multiclase. Es decir, el modelo está prediciendo probabilidades para 4 clases diferentes para cada una de las 6 muestras.


        # Obtener las etiquetas verdaderas para este lote
        target = train_Y[i * batch_size:(i+1)*batch_size]

        #······························································································ADAPTACIÓN A MULTCLASE······························································································

        # PRIMER CAMBIO -----------Aplicar softmax a la salida para convertirla en probabilidades por clase
        #Sin embargo, si estás usando CrossEntropyLoss, no debes aplicar softmax explícitamente. Esta función ya combina el softmax y la entropía cruzada internamente, por lo que puedes eliminar esa línea:
        # output = torch.softmax(output, dim=1)  # Dim=1 para aplicar softmax en las clases, dica que el softmax debe aplicarse a lo largo de las dimensiones de las clases (es decir, a lo largo de cada fila de la salida).

        # 2) Calcular la pérdida
        loss = criterion(output, target)

        # 3) Realizar retropropagación, calcular gradientes
        loss.backward()

        # 4) Actualizar los parámetros del modelo
        optimizer.step()

        #SEGUNDO CAMBIO----------------------------------------------------------------------EN VEZ DE SIGMOID
        # 5)  Calcular precisión para este lote
        # Obtener la clase con la mayor probabilidad para cada muestra, devuelve el índice de la clase con la probabilidad más alta para cada muestra del lote.
        preds = torch.argmax(output, dim=1)

        # Calcular cuántas predicciones fueron correctas
        correct += (preds == target).sum().item() # da un tensor de valores True o False (1 o 0), se suma la cantidad de predicciones correctas usando .sum().item().


        #La pérdida loss se calcula usando la función de pérdida proporcionada (por ejemplo, torch.nn.CrossEntropyLoss() si estás haciendo clasificación multiclase).
        #Como se está utilizando softmax para la salida, asegúrate de usar una CrossEntropyLoss como la función de pérdida, ya que esta función combina tanto el softmax como la entropía cruzada en un solo paso. sisis

        # Sumar la pérdida total
        total_loss += loss.detach()


        #···························································································································································································

        #accuracy = correct / num_samples 

        #calcular la precicisón del lote
        #preds_binary = torch.sigmoid(output) >= 0.5 ########################################
        #target_binary = target >= 0.5

        #Esta parte es sustituida por correct += (preds == target).sum().item().  En clasificación multiclase, ya no estás comparando valores binarios (verdadero/falso) como en clasificación binaria. 
        #           En lugar de eso, debes comparar las predicciones de clase (es decir, los índices de la clase predicha) con las etiquetas verdaderas, que también son índices de clase.
        #for j in range(preds_binary.shape[0]):
        #    if preds_binary[j] == target_binary[j]:
       #         correct += 1
             
    if verbose:    
        print('Train Epoch: {} LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
            epoch, total_loss / num_samples * batch_size, correct, num_samples, 100. * correct / num_samples))
        
    return (total_loss / num_samples * batch_size).item(), (correct / num_samples)

    
def eval_epoch(model, X, Y, criterion, name, verbose=False):
    model.eval()
    correct = 0
    num_samples = Y.shape[0]
    
    output = model(X[0], X[1], X[2]) #.view(-1)
    
    
    #preds_binary = torch.sigmoid(output) >= 0.5
    #target_binary = Y >= 0.5

    # Aplicar softmax a la salida para obtener probabilidades por clase
    #output = torch.softmax(output, dim=1)  # Dim=1 para aplicar softmax en las clases

     # Calcular la pérdida
    loss = criterion(output, Y)

  # Obtener la clase predicha para cada muestra (la clase con mayor probabilidad)
    preds = torch.argmax(output, dim=1)

    # Comparar las predicciones con las etiquetas verdaderas
    correct = (preds == Y).sum().item()
    
   # for j in range(preds_binary.shape[0]):
   #     if preds_binary[j] == target_binary[j]:
    #        correct += 1
            
    if verbose:
        print('{} set: LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)\n'.format(
            name, loss, correct, num_samples, 100. * correct / num_samples))
    return loss.item(), correct / num_samples




#from sklearn.metrics import precision_score, recall_score, f1_score

#def calculate_metrics(predictions, targets):
#    precision = precision_score(targets, predictions, average='weighted')
#   recall = recall_score(targets, predictions, average='weighted')
#    f1 = f1_score(targets, predictions, average='weighted')
#    return precision, recall, f1
