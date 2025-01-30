import statistics 
import pandas as pd
import numpy as np
import heapq
import torch
import copy 
import matplotlib.pyplot as plt

f2_cases = ['wu', 'uw', 'w', 'noop']
f3_cases = ['dr', 'cr', 'r', 'noop']

def get_count_cases(samples_list):
    
    cases_dict = dict.fromkeys([f3 + '_' + f2 for f3 in f3_cases for f2 in f2_cases], 0)
    
    def fill_f2_pattern(cases_dict, f3, sample):
        if sample[1].find('w') != -1 and sample[1].find('u') != -1:
            if sample[1].index('w') < sample[1].index('u'):
                cases_dict[f3 + 'wu'] += 1
            else:
                cases_dict[f3 + 'uw'] += 1
        elif sample[1].find('w') != -1:
            cases_dict[f3 + 'w'] += 1
        else:
            cases_dict[f3 + 'noop'] += 1

    for i in range(len(samples_list)):
        if samples_list[i][2].find('d') != -1 and samples_list[i][2].find('r') != -1:
            fill_f2_pattern(cases_dict, 'dr_', samples_list[i])
        elif samples_list[i][2].find('c') != -1 and samples_list[i][2].find('r') != -1:
            fill_f2_pattern(cases_dict, 'cr_', samples_list[i])
        elif samples_list[i][2].find('r') != -1:
            fill_f2_pattern(cases_dict, 'r_', samples_list[i])
        else:
            fill_f2_pattern(cases_dict, 'noop_', samples_list[i])
    
    return cases_dict

def get_count_case(count_cases, case, f=None):
    if f == 'f3':
        return sum([v for k, v in count_cases.items() if (k.find(case) == 0)])
    elif f == 'f2':
        return sum([v for k, v in count_cases.items() if (k.find(case) == len(k) - len(case))])
    else:
        raise ValueError("Shouldn't get here")
        
def delete_case(count_cases, case, f):
    if f == 'f3':
        keys = [k for k in count_cases.keys() if (k.find(case) == 0)]
        for k in keys:
            count_cases.pop(k, 0)
    elif f == 'f2':
        keys = [k for k in count_cases.keys() if (k.find(case) == len(k) - len(case))]
        for k in keys:
            count_cases.pop(k, 0)
    else:
        raise ValueError("Shouldn't get here")
        

def get_stats_and_ratio(samples_original):

    count_cases = get_count_cases(samples_original)

    print("Number of samples by case:")
    
    for f3 in f3_cases:
        output_str = "F3 " + f3 + ":    "
        
        for f2 in f2_cases:
            output_str += "F2-" + f2 + " " + str(count_cases[f3 + '_' + f2]) + ' | '
        
        print(output_str)
        
    print()
    
    pos_samples_count = get_count_case(count_cases, 'r', 'f3') + count_cases['dr_uw'] + count_cases['dr_w'] + count_cases['dr_noop']
    print('Positive samples count:', pos_samples_count)
    print('Total samples count:', len(samples_original))

    pos_ratio = pos_samples_count / len(samples_original)
    print('Positive class ratio:', pos_ratio)
    return pos_ratio
    
def get_accuracy_by_cases(model, X, Y, original_test_samples):
    model.eval()

    output = model(X[0], X[1], X[2])
    y_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

        
    samples_correct = []

    for i in range(len(y_pred)):
        if y_pred[i] == Y[i]:
            samples_correct.append(original_test_samples[i])

    preds = get_count_cases(samples_correct)
    total = get_count_cases(original_test_samples)
    
    accuracies = {}
    
    for k, v in preds.items():
        accuracies[k] = preds[k] / total[k]
        
    accuracies['Overall'] = len(samples_correct) / len(original_test_samples)
            
    return accuracies

def get_precision_by_cases(model, X, Y, original_test_samples): #Precisión = VP / (VP + FP), la precisión mide cuántas de las predicciones positivas realmente lo son
    model.eval()

    output = model(X[0], X[1], X[2])
    y_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
    #Podrian ser contadores, pero se queda asi para poder debbugear
    v0=[]
    f0=[]
    v1=[]
    f1=[]
    v2=[]
    f2=[]
    v3=[]
    f3=[]
    
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if Y[i] == 0:v0.append(original_test_samples[i])
            else:f0.append(original_test_samples[i]) #falsos positivos      
        elif y_pred[i] == 1:
            if Y[i] == 1:v1.append(original_test_samples[i])
            else:f1.append(original_test_samples[i]) #falsos positivos
        elif y_pred[i] == 2:
            if Y[i] == 2:v2.append(original_test_samples[i])
            else:f2.append(original_test_samples[i]) #falsos positivos
        elif y_pred[i] == 3:
            if Y[i] == 3:v3.append(original_test_samples[i])
            else:f3.append(original_test_samples[i]) #falsos positivos

    if len(v0)+len(f0) > 0: precision0 = len(v0)/(len(v0)+len(f0))
    else: precision0 = 0.0
    
    if len(v1)+len(f1) > 0: precision1 = len(v1)/(len(v1)+len(f1))
    else: precision1 = 0.0

    if len(v2)+len(f2)>0: precision2 = len(v2)/(len(v2)+len(f2))
    else: precision2 = 0.0

    if len(v3)+len(f3)>0: precision3 = len(v3)/(len(v3)+len(f3))
    else: precision3 = 0.0

    # Calcular precisión por cada combinación f3_f2
    precision = {}
    #Para calcular la precisión por cada clase
    precision['A'] = precision0
    precision['D'] = precision1
    precision['R'] = precision2
    precision['V'] = precision3

    # Calcular precisión general
    precision['Overall'] = (precision0+precision1+precision2+precision3) / 4 #media aritmetica o media ponderada?

    return precision

def get_recall_by_cases(model, X, Y, original_test_samples): #VP/VP+FN
    model.eval()

    # Obtener las predicciones del modelo
    output = model(X[0], X[1], X[2])
    y_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

    #Podrian ser contadores, pero se queda asi para poder debbugear
    v0=[]
    f0=[]
    v1=[]
    f1=[]
    v2=[]
    f2=[]
    v3=[]
    f3=[]
    
    for i in range(len(y_pred)):
        
        if Y[i] == 0:
            if y_pred[i] == 0:v0.append(original_test_samples[i]) 
            else:f0.append(original_test_samples[i])   
        elif y_pred[i] == 1:
            if Y[i] == 1:v1.append(original_test_samples[i])
            else:f1.append(original_test_samples[i]) #falsos positivos
        elif Y[i] == 2:
            if y_pred[i] == 2:v2.append(original_test_samples[i])
            else:f2.append(original_test_samples[i]) #falsos positivos
        elif Y[i] == 3:
            if y_pred[i] == 3:v3.append(original_test_samples[i])
            else:f3.append(original_test_samples[i]) #falsos positivos
        

    if len(v0)+len(f0) > 0: recall0 = len(v0)/(len(v0)+len(f0))
    else: recall0 = 0.0
    
    if len(v1)+len(f1) > 0: recall1 = len(v1)/(len(v1)+len(f1))
    else: recall1 = 0.0

    if len(v2)+len(f2)>0: recall2 = len(v2)/(len(v2)+len(f2))
    else: recall2 = 0.0

    if len(v3)+len(f3)>0: recall3 = len(v3)/(len(v3)+len(f3))
    else: recall3 = 0.0

    recalls={}
    
    #Para calcular el recall por cada clase
    recalls['A'] = recall0
    recalls['D'] = recall1
    recalls['R'] = recall2
    recalls['V'] = recall3

    
    recalls['Overall'] = (recall0+recall1+recall2+recall3) / 4 #media aritmetica o media ponderada?

    return recalls

def get_f1_by_cases(precision, recall):

    f1_scores={}
    
    # Calcular el F1-score general (Overall)
    f1_scores['Overall'] = (
        2 * (precision['Overall'] * recall['Overall']) / (precision['Overall'] + recall['Overall'])
        if (precision['Overall'] + recall['Overall']) > 0 else 0.0
    )
    
    return f1_scores

def imprimirMetricas(model_names, all_accuracies, all_precisions, all_recalls, all_f1_scores):
    
    # Lista para almacenar los datos
    model_metrics = []

    for i, model_name in enumerate(model_names):
        # Esta linea sirve para encontrar el índice del episodio con el mejor accuracy
        best_index = max(range(len(all_accuracies[i])), key=lambda idx: all_accuracies[i][idx]['Overall'])

        # Obtenemos las métricas del mejor episodio 
        best_accuracy = all_accuracies[i][best_index]['Overall']
        best_precision = all_precisions[i][best_index]['Overall']
        best_recall = all_recalls[i][best_index]['Overall']
        best_f1_score = all_f1_scores[i][best_index]['Overall']

        # Obtenemos las métricas por clase del mejor episodio
        best_precision_by_class = {cls: all_precisions[i][best_index][cls] for cls in ['A', 'D', 'R', 'V']}
        best_recall_by_class = {cls: all_recalls[i][best_index][cls] for cls in ['A', 'D', 'R', 'V']}

        # Guardar en una lista para el DataFrame
        row = {
            'Modelo': model_name,
            'Accuracy': best_accuracy,
            'Precision': best_precision,
            'Recall': best_recall,
            'F1-score': best_f1_score
        }

        for cls in ['A', 'D', 'R', 'V']:
            row[f'Precision clase {cls}'] = best_precision_by_class[cls]
            row[f'Recall clase {cls}'] = best_recall_by_class[cls]

        model_metrics.append(row)

    # Crear DataFrame
    df_metrics = pd.DataFrame(model_metrics)
    #Esta linea deberia imprimir el df pero no lo hace, si se quiere imprimir desde aqui, aunq con peor formato, solo añadir print()
    df_metrics
    
def decodificar(a):
    ret=["A", "D", "R", "V"]
    return ret[a]
def convert_to_percentage(x):
    return str(round(x * 100, 1)) + '%'

def get_stats(accuracies, display_stdev):
    if display_stdev:
        stats = [(statistics.mean(np.array(accuracies)[:,j]), statistics.stdev(np.array(accuracies)[:,j])) for j in range(len(accuracies[0]))]
        return [(convert_to_percentage(x), ' ' + convert_to_percentage(y)) for (x, y) in stats]
    else:
        return [convert_to_percentage(statistics.mean(np.array(accuracies)[:,j])) for j in range(len(accuracies[0]))]
    
def get_stats_df(models_accuracies, model_names, original_test_samples, collapsed_ops=[], display_stdev=False):
    
    sample_count = get_count_cases(original_test_samples)

    models_accuracies_cpy = copy.deepcopy(models_accuracies)
        
    def collapse(model_acc, f3_op):
        
        avg_acc = 0
        
        for f2_op in f2_cases:
            avg_acc += model_acc[f3_op + '_' + f2_op] * sample_count[f3_op + '_' + f2_op]
            
        model_acc['f3_' + f3_op] = avg_acc / get_count_case(sample_count, f3_op, 'f3')
        delete_case(model_acc, f3_op, 'f3')
                
    for model_accuracies in models_accuracies_cpy:
        for model_accuracy in model_accuracies:
            for collapsed_op in collapsed_ops:
                collapse(model_accuracy, collapsed_op)
                
    for collapsed_op in collapsed_ops:
        sample_count['f3_' + collapsed_op] = get_count_case(sample_count, collapsed_op, 'f3')
        delete_case(sample_count, collapsed_op, 'f3')
        
    col_names = sorted(models_accuracies_cpy[0][0].keys())
        
    stats = []
    for model_accuracies in models_accuracies_cpy:
        val_sorted_by_keys = []
        for model_accuracy in model_accuracies:
            val_sorted_by_keys += [[snd for (fst, snd) in sorted(model_accuracy.items())]]
            
        stats.append(get_stats(val_sorted_by_keys, display_stdev))

    stats.append([len(original_test_samples)] + [snd for (fst, snd) in sorted(sample_count.items())])
    
    results_pd = pd.DataFrame(stats, columns=col_names)
    
    row_names = {}
    for i in range(len(model_names)):
         row_names[i] = model_names[i]
    row_names[len(model_names)] = 'Sample count'
    results_pd.rename(index=row_names)
    return results_pd.rename(index=row_names)



def reshape_1_sample(x_test_f1, x_test_f2, x_test_f3):
    if len(x_test_f1.shape) > 1:
        # convolutional or lstm
        return [x_test_f1.reshape((1, x_test_f1.shape[0], x_test_f1.shape[1])),
                x_test_f2.reshape((1, x_test_f2.shape[0], x_test_f1.shape[1])),
                x_test_f3.reshape((1, x_test_f3.shape[0], x_test_f1.shape[1]))]
    else:
        # deepset / concat
        return [x_test_f1.reshape((1, x_test_f1.shape[0])),
                x_test_f2.reshape((1, x_test_f2.shape[0])),
                x_test_f3.reshape((1, x_test_f3.shape[0]))]

def get_wrong_predictions(model, x_test, y_test, original_test_samples):
    model.eval()
    
    fa = []
    fd = []
    fr = []
    fv = []
    
    output = model(x_test[0], x_test[1], x_test[2]) #Al parecer .view(-1) hace que la salida se aplane en una sola dimension o algo asi 
    y_pred = torch.argmax(output, dim=1)  # Clasificacion Multiclase
    
    y_test = y_test.long()  #Nos aseguramos que y_test es del tipo correcto, ya que tiene que ser long para la funcion de perdida que tenemos creo
     
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if y_test[i] != 0:fa.append((original_test_samples[i],decodificar(y_pred[i]))) #falsos atomicos      
        elif y_pred[i] == 1:
            if y_test[i] != 1:fd.append((original_test_samples[i],decodificar(y_pred[i]))) #falsos positivos
        elif y_pred[i] == 2:
            if y_test[i] != 2:fr.append((original_test_samples[i],decodificar(y_pred[i]))) #falsos positivos
        elif y_pred[i] == 3:
            if y_test[i] != 3:fv.append((original_test_samples[i],decodificar(y_pred[i]))) #falsos positivos

    return fa, fd, fr, fv

def get_wrong_predictions_bycases(model, x_test, y_test, original_test_samples):
    model.eval()
    
    fa = []
    fd = []
    fr = []
    fv = []
    
    output = model(x_test[0], x_test[1], x_test[2]) #Al parecer .view(-1) hace que la salida se aplane en una sola dimension o algo asi 
    y_pred = torch.argmax(output, dim=1)  # Clasificacion Multiclase
    
    y_test = y_test.long()  #Nos aseguramos que y_test es del tipo correcto, ya que tiene que ser long para la funcion de perdida que tenemos creo
     
    for i in range(len(y_pred)):
        if y_test[i] == 0:
            if y_pred[i] != 0:fa.append((original_test_samples[i],decodificar(y_pred[i]))) #Fallos en casos de atomicidad      
        elif y_test[i] == 1:
            if y_pred[i] != 1:fd.append((original_test_samples[i],decodificar(y_pred[i]))) #Fallos en casos de condicion de carrera
        elif y_test[i] == 2:
            if y_pred[i] != 2:fr.append((original_test_samples[i],decodificar(y_pred[i]))) #Fallos en casos de DeadLock
        elif y_test[i] == 3:
            if y_pred[i] != 3:fv.append((original_test_samples[i],decodificar(y_pred[i]))) #Fallos en casos Validos

    return fa, fd, fr, fv

def filter_top_k_accuracies(accuracies, top_k):
    return heapq.nlargest(top_k, accuracies, key=lambda x: x['Overall'])

def print_wrong_preds(wrong_preds_list, k=10):
        a=0
        cases=["Atomicity violation", "DeadLock", "Data race ", "Valid"]
        for i in wrong_preds_list:
            
            print(f'{min(k,len(i))} false {cases[a]}:')   
            for j in range(min(k,len(i))):
                print(f"Sample {i[j][0]} | Prediction {i[j][1]}")
            a+=1
            print("\n")

def print_wrong_preds_bycases(wrong_preds_list, k=10):
        a=0
        cases=["Atomicity violation", "DeadLock", "Data race", "Valid"]
        for i in wrong_preds_list:
            
            print(f'{min(k,len(i))} {cases[a]} cases wrong predicted:')   
            for j in range(min(k,len(i))):
                print(f"Sample {i[j][0]} | Prediction {i[j][1]}")
            a+=1
            print("\n")
        
def get_stats_per_model(model_accuracies, model_names, original_test_samples, collapsed_ops=[]):
    
    sample_count = get_count_cases(original_test_samples)

    model_accuracies_cpy = copy.deepcopy(model_accuracies)
        
    def collapse(model_acc, f3_op):
        
        avg_acc = 0
        
        for f2_op in f2_cases:
            avg_acc += model_acc[f3_op + '_' + f2_op] * sample_count[f3_op + '_' + f2_op]
            
        model_acc['f3_' + f3_op] = avg_acc / get_count_case(sample_count, f3_op, 'f3')
        delete_case(model_acc, f3_op, 'f3')
                
    for model_accuracy in model_accuracies_cpy:
        for collapsed_op in collapsed_ops:
            collapse(model_accuracy, collapsed_op)
                
    for collapsed_op in collapsed_ops:
        sample_count['f3_' + collapsed_op] = get_count_case(sample_count, collapsed_op, 'f3')
        delete_case(sample_count, collapsed_op, 'f3')
        
    col_names = sorted(model_accuracies_cpy[0].keys())
        
    stats = []
    for model_accuracy in model_accuracies_cpy:
        stats.append([convert_to_percentage(val) for (key, val) in sorted(model_accuracy.items())]) 

    stats.append([len(original_test_samples)] + [snd for (fst, snd) in sorted(sample_count.items())])
    
    results_pd = pd.DataFrame(stats, columns=col_names)
    
    row_names = {}
    for i in range(len(model_names)):
         row_names[i] = model_names[i]
    row_names[len(model_names)] = 'Sample count'
    results_pd.rename(index=row_names)
    return results_pd.rename(index=row_names)
        
def wrong_preds_separate_to_merged(wrong_preds):
    wrong_preds_cpy = [fst + snd for (fst, snd) in wrong_preds.copy()]
    for i in range(len(wrong_preds_cpy)):
        wrong_preds_cpy[i] = [fst for fst, snd in wrong_preds_cpy[i]]
        
    return wrong_preds_cpy
    
def print_aggregate_wrong_preds(wrong_preds_FP_FN, model_names, collapsed_ops=[]):
    
    wrong_preds_merged = copy.deepcopy(wrong_preds_separate_to_merged(wrong_preds_FP_FN))
                
    for i in range(len(wrong_preds_merged)):
        wrong_preds_merged[i] = get_count_cases(wrong_preds_merged[i])
        
        for collapsed_op in collapsed_ops:
            wrong_preds_merged[i]['f3_' + collapsed_op] = get_count_case(wrong_preds_merged[i], collapsed_op, 'f3')
            delete_case(wrong_preds_merged[i], collapsed_op, 'f3')          
        
    col_names = sorted(wrong_preds_merged[i].keys())
        
    for i in range(len(wrong_preds_merged)):
        wrong_preds_merged[i] = [val for (key, val) in sorted(wrong_preds_merged[i].items())]
    
    results_pd = pd.DataFrame(wrong_preds_merged, columns=col_names)
    
    row_names = {}
    for i in range(len(model_names)):
         row_names[i] = model_names[i]
    row_names[len(model_names)] = 'Sample count'
    results_pd.rename(index=row_names)
    return results_pd.rename(index=row_names)


def display_epochs_stats(epoch_stats, num_experiments, display_train_loss=True,
                         display_val_loss=True, display_train_acc=True, display_val_acc=True, ncols=3):
    def plot_single(axes, index):
        x = np.arange(len(epoch_stats[0][index]))

        line_names = []
        if display_train_loss:
            axes.plot(x, epoch_stats[0][index], color='cyan')
            line_names.append(['Train loss'])
        if display_val_loss:
            axes.plot(x, epoch_stats[1][index], color='blue')
            line_names.append(['Val loss'])
        if display_train_acc:
            axes.plot(x, epoch_stats[2][index], color='pink')
            line_names.append(['Train acc'])
        if display_val_acc:
            axes.plot(x, epoch_stats[3][index], color='red')
            line_names.append(['Val acc'])

        axes.legend(line_names, loc='upper right')
        if(hasattr(axes, 'set_xlabel')):
            axes.set_xlabel("Number of epochs")
            axes.set_title("Epoch stats experiment #" + str(index))
        else:
            axes.xlabel("Number of epochs")
            axes.title("Epoch stats experiment #" + str(index))
        
    aux = 1 if (num_experiments % ncols)!=0 else 0
    for i in range(int(num_experiments / ncols) + aux):
        
        num_figs = ncols if i < int(num_experiments / ncols) else num_experiments % ncols
        fig, axes = plt.subplots(nrows=1, ncols=num_figs, figsize=(16,4))
        
        if num_figs == 1:
            plot_single(plt, i * ncols)
        else:
            for j in range(num_figs):
                index = i * ncols + j
                plot_single(axes[j], index)
            
        fig.tight_layout()