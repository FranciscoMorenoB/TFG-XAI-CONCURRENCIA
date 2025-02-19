import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Data():

    def random_no_op(self):
        noops = [',', '.', '_']
        dice = self.Random.randint(0, 2)
        return noops[dice]

    def generate_f(self, f_op, pos, interop_dist):
        f = ''

        for k in range(self.layer_size):
            
            if pos == k and len(f_op) > 0:
                f += f_op[0]
            elif k == pos + interop_dist + 1 and len(f_op) > 1:
                f += f_op[1]
            else:
                f += self.random_no_op()
                
        return f

        
    def is_buggy(self, sample):
        if 'r' in sample[2] and 'd' not in sample[2] and 'c' not in sample[2]:
            return True
        if 'd' in sample[2]:
            if 'w' in sample[1] and 'u' in sample[1]:
                return sample[1].index('w') >= sample[1].index('u')
            else:
                return True
        return False
    
    def tipe_buggy(self, sample):
        if (('r' not in sample[2] and 'd' not in sample[2] and 'c' not in sample[2]) or 'c' in sample[2] and 'r' in sample[2]): return "V"
        if ('d' not in sample[2] and'r' in sample[2]): return "R"
        if ('u' in sample[1]): #no hace falta comprobar que en sample2 hay dr
            if sample[1].index('w') >= sample[1].index('u') : return "A"
            else: return "V"
        else: return "D"
        
    
    def is_buggy2(self, sample):
        if (('r' not in sample[2] and 'd' not in sample[2] and 'c' not in sample[2]) or 'c' in sample[2] and 'r' in sample[2]): return True
        if ('r' in sample[2]): return False
        if ('u' in sample[1]): #no hace falta comprobar que en sample2 hay dr
            if sample[1].index('w') >= sample[1].index('u') : return False
            else: return True
        return False
    
    
    def generate_samples(self):
        
        samples = []
        
        interval_factor = self.permutation_intervals
        
        self.first_interval_size = 0
        self.last_interval_size = 0
        
        cnt = 0
        
        for k_i in range(interval_factor):
            for k_j in range(interval_factor):
                for f2_op in self.f2_operations:
                    for f3_op in self.f3_operations:
                        for interop_dist in self.interop_distances: 
                            i_start = int(((self.layer_size - len(f2_op) - interop_dist + 1) / interval_factor) * k_i)
                            i_limit = int(((self.layer_size - len(f2_op) - interop_dist + 1) / interval_factor) * (k_i + 1))

                            j_start = int(((self.layer_size - len(f3_op) - interop_dist + 1) / interval_factor) * k_j)
                            j_limit = int(((self.layer_size - len(f3_op) - interop_dist + 1) / interval_factor) * (k_j + 1))

                            for i in range(i_start, i_limit):
                                for j in range(j_start, j_limit):
                                    f1 = self.generate_f('', 0, -self.layer_size - 1)
                                    f2 = self.generate_f(f2_op, i, interop_dist)
                                    f3 = self.generate_f(f3_op, j, interop_dist)

                                    samples.append([f1, f2, f3, self.tipe_buggy([f1, f2, f3])])
                                    cnt += 1
                                    
                if self.first_interval_size == 0:
                    self.first_interval_size = cnt
                    
                self.last_interval_size = cnt
                cnt = 0
                                
        return samples
        
    def separate_string_chars(self, samples):
        samples_char_separate = []
            
        for i in range(len(samples)):
            samples_char_separate.append([])
            for j in range(len(samples[i]) - 1):
                for k in range(len(samples[i][j])):
                    samples_char_separate[i].append(samples[i][j][k])
            samples_char_separate[i].append(samples[i][-1])
    
        return samples_char_separate
    
    def one_hot_encode(self, samples_char_sep):
        df = pd.DataFrame(samples_char_sep,  columns=self.feature_names+['Correct'])
        unique_chars = set()

        for col in df:
            unique_chars = unique_chars.union(df[col].unique())
       # unique_chars.remove(True)
       # unique_chars.remove(False)
        unique_chars.remove('A')
        unique_chars.remove('R')
        unique_chars.remove('D')
        unique_chars.remove('V')
        unique_chars = sorted(list(unique_chars))

        self.le = LabelEncoder()
        self.le.fit(unique_chars)

        self.feature_names_one_hotted  = []

        for i in range(3):
            for j in range(self.layer_size):
                for c in self.le.classes_:
                    self.feature_names_one_hotted .append(f'f{i + 1}-{j}-{c}') #f{número de función}-{posición}-{carácter}.

        df_no_label = df.drop(['Correct'], axis=1)
        df_encoded = df_no_label.apply(self.le.transform)

        unique_labels = set()

        for col in df_encoded:
            unique_labels = unique_labels.union(df_encoded[col].unique())
        df_unique_labels = pd.DataFrame(unique_labels)

        for i in range(1, 3 * self.layer_size):
            df_unique_labels[i] = df_unique_labels[0]

        self.unique_labels = df_unique_labels

        self.ohe = OneHotEncoder(categories='auto')
        self.ohe.fit(df_unique_labels)

        one_hotted_df = pd.DataFrame(self.ohe.transform(df_encoded).toarray(), columns=self.feature_names_one_hotted )
        self.num_one_hot_encodings = int(one_hotted_df.shape[1] / self.layer_size / 3)
        one_hotted_df['label'] = df['Correct']

        columns_one_hotted = list(one_hotted_df.columns.values[:-1])

        self.f1_start = 0

        self.f2_start = self.f1_start
        while (columns_one_hotted[self.f2_start].find('f2') == -1):
            self.f2_start += 1

        self.f3_start = self.f2_start
        while (columns_one_hotted[self.f3_start].find('f3') == -1):
            self.f3_start += 1

        self.f3_end = len(columns_one_hotted)

        return np.array(one_hotted_df)

    def get_one_hot_encoding(self, samples_char_sep, unique_labels=None):
        df = pd.DataFrame(samples_char_sep)
        
        df_encoded = df.apply(self.le.transform)
        
        if unique_labels is None:
            return self.ohe.transform(df_encoded).toarray()
        else:
            ohe = OneHotEncoder(categories='auto')
            ohe.fit(unique_labels)
            
            return ohe.transform(df_encoded).toarray()
        
    def generate_one_hotted_noops(self, count):
        if count == 0:
            return np.zeros((0, self.num_one_hot_encodings))
                            
        noops = []
        for i in range(count):
            noops.append(self.random_no_op())
        return self.get_one_hot_encoding(noops, self.unique_labels.loc[:,:0])
    
    def reverse_encoding(self, nparray):
        reverse_ohe = self.ohe.inverse_transform(nparray)
        reverse_le = np.apply_along_axis(self.le.inverse_transform, 0, reverse_ohe)
        reverse_separate = np.apply_along_axis(''.join, 1, reverse_le)
        undivide = lambda x: [x[:self.layer_size], x[self.layer_size:self.layer_size*2],  x[self.layer_size*2:], \
                              self.tipe_buggy([x[:self.layer_size], x[self.layer_size:self.layer_size*2],  x[self.layer_size*2:]])]
        reverse_divide = list(map(undivide, list(reverse_separate)))
        return reverse_divide
    
    def get_splits_subsampled(self, data, split_ratios):
        
        assert(split_ratios[0] + split_ratios[1] == 1)
        
        cur_data = data.copy()
        np.random.shuffle(cur_data)
        split = int(cur_data.shape[0] * split_ratios[0])
        return cur_data[:split], cur_data[split:] 
        
    def get_splits_skipped(self, data, step_size_train):
        
        assert (step_size_train >= 2 and isinstance(step_size_train, int))
        
        train_data = []
        rest_data = []

        for i in range(data.shape[0]):
            if i % step_size_train == 0:
                train_data.append(data[i])
            else:
                rest_data.append(data[i])

        train_data = np.array(train_data)
        rest_data = np.array(rest_data)
        
        return train_data, rest_data
    
    def get_two_ops_distance(self, sample):
        rev = self.reverse_encoding(np.array([sample[:-1]]))[0]
        if 'c' in rev[2]:
            return rev[2].index('r') - rev[2].index('c') - 1
        elif 'd' in rev[2]:
            return rev[2].index('r') - rev[2].index('d') - 1
        elif 'u' in rev[1]:
            return abs(rev[1].index('u') - rev[1].index('w')) - 1
        return -1
    
    def get_splits_omit_distances(self, data, omit_distances):
        
        train_data = []
        rest_data = []

        for i in range(data.shape[0]):
            if self.get_two_ops_distance(data[i]) not in omit_distances: 
                train_data.append(data[i])
            else:
                rest_data.append(data[i])

        train_data = np.array(train_data)
        rest_data = np.array(rest_data)
        
        return train_data, rest_data
    
    def get_splits_1st_interval(self, data):
        return data[:self.first_interval_size], data[-self.last_interval_size:]
    
    def get_splits(self, modes, parameters, val_test_split=0.2):
        
        param_ind = 0
        train_data = self.np_data
        rest_data = None
        
        for mode in modes:
            if mode == 'random_subsample':
                split_ratios = parameters[param_ind]
                param_ind += 1
                                
                new_train, new_rest = self.get_splits_subsampled(train_data, split_ratios)
            elif mode == 'skip':
                step_size = parameters[param_ind]
                param_ind += 1
                
                new_train, new_rest = self.get_splits_skipped(train_data, step_size)
            elif mode == '1st_interval':
                new_train, new_rest = self.get_splits_1st_interval(train_data)
            elif mode == 'omit_distances':
                omit_distances = parameters[param_ind]
                param_ind += 1
                
                new_train, new_rest = self.get_splits_omit_distances(train_data, omit_distances)
            train_data = new_train
            if rest_data is None:
                rest_data = new_rest
            else:
                rest_data = np.concatenate((rest_data, new_rest))
        
        if 'omit_distances' in modes:
            rest_data, _ = self.get_splits_omit_distances(rest_data, list(set(self.interop_distances) - set(omit_distances)))
        
        val_data, test_data = self.get_splits_subsampled(rest_data, [val_test_split, 1-val_test_split])
                
        print(f'Number of samples: {train_data.shape[0]} train | {val_data.shape[0]} val | {test_data.shape[0]} test')
        return train_data, val_data, test_data
        
    def get_x_y(self, np_data):
        return np_data[:,:-1], np_data[:,-1]
    
    def to_conv_format(self, x, padding_left_size=0, padding_right_size=0):
        x_f1 = x[:,self.f1_start:self.f2_start].copy().reshape(x.shape[0], self.layer_size * self.num_one_hot_encodings, 1)
        x_f2 = x[:,self.f2_start:self.f3_start].copy().reshape(x.shape[0], self.layer_size * self.num_one_hot_encodings, 1)
        x_f3 = x[:,self.f3_start:].copy().reshape(x.shape[0], self.layer_size * self.num_one_hot_encodings, 1)

        x_f1 = []
        x_f2 = []
        x_f3 = []

        for i in range(x.shape[0]):

            f1 = x[i][self.f1_start:self.f2_start].copy().reshape(self.layer_size * self.num_one_hot_encodings)
            f2 = x[i][self.f2_start:self.f3_start].copy().reshape(self.layer_size * self.num_one_hot_encodings)
            f3 = x[i][self.f3_start:].copy().reshape(self.layer_size * self.num_one_hot_encodings)
                            
            get_padding = self.generate_one_hotted_noops
            
            x_f1.append(np.concatenate((get_padding(padding_left_size).reshape((-1)), f1,
                                        get_padding(padding_right_size).reshape((-1)))))
            x_f2.append(np.concatenate((get_padding(padding_left_size).reshape((-1)), f2,
                                        get_padding(padding_right_size).reshape((-1)))))
            x_f3.append(np.concatenate((get_padding(padding_left_size).reshape((-1)), f3,
                                        get_padding(padding_right_size).reshape((-1)))))
                        
                        
        x_f1 = np.array(x_f1)
        x_f2 = np.array(x_f2)
        x_f3 = np.array(x_f3)
        
        return [torch.from_numpy(x_f1.astype(np.float32)).float(), 
                torch.from_numpy(x_f2.astype(np.float32)).float(),
                torch.from_numpy(x_f3.astype(np.float32)).float()]

        
    def to_lstm_format(self, x, padding_left_size=0, padding_right_size=0):
        x_f1 = []
        x_f2 = []
        x_f3 = []

        for i in range(x.shape[0]):

            f1 = x[i][self.f1_start:self.f2_start].copy().reshape(self.layer_size, self.num_one_hot_encodings)
            f2 = x[i][self.f2_start:self.f3_start].copy().reshape(self.layer_size, self.num_one_hot_encodings)
            f3 = x[i][self.f3_start:].copy().reshape(self.layer_size, self.num_one_hot_encodings)
                            
            get_padding = self.generate_one_hotted_noops
            
            x_f1.append(np.concatenate((get_padding(padding_left_size), f1, get_padding(padding_right_size))))
            x_f2.append(np.concatenate((get_padding(padding_left_size), f2, get_padding(padding_right_size))))
            x_f3.append(np.concatenate((get_padding(padding_left_size), f3, get_padding(padding_right_size))))
                        
                        
        x_f1 = np.array(x_f1)
        x_f2 = np.array(x_f2)
        x_f3 = np.array(x_f3)
                        
#         x_f1 = x[:,self.f1_start:self.f2_start].copy().reshape(x.shape[0], self.layer_size, self.num_one_hot_encodings)
#         x_f2 = x[:,self.f2_start:self.f3_start].copy().reshape(x.shape[0], self.layer_size, self.num_one_hot_encodings)
#         x_f3 = x[:,self.f3_start:].copy().reshape(x.shape[0], self.layer_size, self.num_one_hot_encodings)

#         if padding_left > 0:
#             padding = np.zeros((padding_left, self.num_one_hot_encodings))
            
#             for i in range(x_f2.shape[0]):
#                 x_f2[i] = np.concatenate((padding, x_f2[i]))
#                 x_f3[i] = np.concatenate((padding, x_f3[i]))
                
#         if padding_right > 0:
#             padding = np.zeros((padding_right, self.num_one_hot_encodings))
            
#             for i in range(x_f2.shape[0]):
#                 x_f2[i] = np.concatenate((x_f2[i], padding))
#                 x_f3[i] = np.concatenate((x_f3[i], padding))
                                         
        return [torch.from_numpy(x_f1.astype(np.float32)).float(), 
                torch.from_numpy(x_f2.astype(np.float32)).float(),
                torch.from_numpy(x_f3.astype(np.float32)).float()]
    
    def npfloat_to_tensor(self, ndarray):
        return torch.from_numpy(ndarray.astype(np.float32)).float()
    
    def __init__(self, layer_size=4, interop_distances=[0], permutation_intervals=1, seed=777):
        
        self.Random = random.Random(seed)
        np.random.seed(seed)
        
        self.interop_distances = interop_distances
        
        self.f1_operations = ['']
        self.f2_operations = ['w', 'wu', 'uw', '']
        self.f3_operations = ['cr', 'dr', 'r', '']
    
        self.layer_size = layer_size
        self.permutation_intervals = permutation_intervals
        self.feature_names = [f'F{f}_{num}' for f in range(1, 4) for num in range(0, self.layer_size)]
        
        self.samples = self.generate_samples()
        
        self.samples_char_sep = self.separate_string_chars(self.samples)
        
        self.np_data = self.one_hot_encode(self.samples_char_sep)      