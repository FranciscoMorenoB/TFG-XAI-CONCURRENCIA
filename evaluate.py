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

    output = model(X[0], X[1], X[2]).view(-1)
    y_pred = torch.sigmoid(output) >= 0.5
    Y = Y >= 0.5
        
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

def get_fp_fn_indexes(model, x_test, y_test, original_test_samples):
    model.eval()
    
    wrong_preds = []
    fp_index = []
    fn_index = []
    right_preds = []
    
    output = model(x_test[0], x_test[1], x_test[2]).view(-1)
    y_pred = output >= 0.5

    for i in range(len(y_pred)):
        if y_pred[i] and not y_test[i]:
            wrong_preds.append(original_test_samples[i])
            fp_index.append(i)
        elif not y_pred[i] and y_test[i]:
            wrong_preds.append(original_test_samples[i])
            fn_index.append(i)
        else:
            right_preds.append(original_test_samples[i])

    return wrong_preds, fp_index, fn_index, right_preds

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
    
    wrong_preds, fp_index, fn_index, right_preds = get_fp_fn_indexes(model, x_test, y_test, original_test_samples)
        
    wrong_pred_index = sorted(fp_index + fn_index)
    
    x_test_FP = np.array([[x_test[0][i].cpu().numpy(), x_test[1][i].cpu().numpy(), x_test[2][i].cpu().numpy()] for i in fp_index])
    x_test_FN = np.array([[x_test[0][i].cpu().numpy(), x_test[1][i].cpu().numpy(), x_test[2][i].cpu().numpy()] for i in fn_index])
        
    false_positives = []
    for i in fp_index:
        x_test_i = reshape_1_sample(x_test[0][i], x_test[1][i], x_test[2][i])
        output = torch.sigmoid(model(x_test_i[0], x_test_i[1], x_test_i[2]).view(-1))
        pred = [y.item() for y in output]
        false_positives.append((wrong_preds[wrong_pred_index.index(i)], pred))
            
    false_negatives = []
    for i in fn_index:
        x_test_i = reshape_1_sample(x_test[0][i], x_test[1][i], x_test[2][i])
        output = torch.sigmoid(model(x_test_i[0], x_test_i[1], x_test_i[2]).view(-1))
        pred = [y.item() for y in output]
        false_negatives.append((wrong_preds[wrong_pred_index.index(i)], pred))

    return false_positives, false_negatives

def filter_top_k_accuracies(accuracies, top_k):
    return heapq.nlargest(top_k, accuracies, key=lambda x: x['Overall'])

def print_wrong_preds(wrong_preds_list, top_k=10):

    for i in range(len(wrong_preds_list)):
        FPs, FNs = wrong_preds_list[i]
        
        top_k_FPs = heapq.nlargest(top_k, FPs, key=lambda x: x[1][0])
        top_k_FNs = heapq.nlargest(top_k, FNs, key=lambda x: -x[1][0])
        
        def pretty_print(preds):
            for pred in preds:
                print(pred[0][1], pred[0][2], '| label:', pred[0][3], '| actual prediction:', pred[1][0])
                
        print(f'Top {top_k} false positives: \n')
        pretty_print(top_k_FPs)

        print(f'\nTop {top_k} false negatives: \n')
        pretty_print(top_k_FNs)
        
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
    
    for i in range(int(num_experiments / ncols) + 1):
        
        num_figs = ncols if i < int(num_experiments / ncols) else num_experiments % ncols
        fig, axes = plt.subplots(nrows=1, ncols=num_figs, figsize=(16,4))
        
        if num_figs == 1:
            plot_single(plt, i * ncols)
        else:
            for j in range(num_figs):
                index = i * ncols + j
                plot_single(axes[j], index)
            
        fig.tight_layout()
