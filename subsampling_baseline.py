import numpy as np
import pandas as pd
import pickle
from utils import apply_spline_transformation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utils import get_baseline_predictions

'''This file trains baseline models (from the Otto Ettala et al. “Individualised non-contrast 
MRI-based risk estimation and shared decision-making in men with a suspicion of prostate cancer: 
Protocol for multicentre randomised controlled trial (multi-IMPROD V. 2.0)") with subsampling'''

def run_baseline_sub_sampling(silos, n_repeats=1, initial_sample_size=50,
                                                   step_size=25, seed=1234, directory = 'results'):

    np.random.seed(seed)

    auc_results_total = defaultdict(list) # Stores {dataset_name: [{sample_size: auc}, ...]}


    for repeat in tqdm(range(n_repeats)):
        random_state = np.random.randint(0, 10000000)
        for dataset_name, dataset in silos.items():
            
            auc_for_sample_sizes = {}  # {sample_size: auc} for this repetition
            
            # current_sample_size = initial_sample_size
            # max_sample_size = min(len(dataset), 600)

            # while current_sample_size <= max_sample_size:
            #     test_fraction = current_sample_size / max_sample_size
            #     if test_fraction < 1:
            #         train_df, test_df = train_test_split(dataset, test_size=test_fraction, stratify=dataset['sig_cancer'], random_state=random_state)
            #     else:
            #         test_df = dataset
                
            #     test_df['pred'] = dataset.apply(get_baseline_predictions, axis=1)
            #     auc = roc_auc_score(test_df['sig_cancer'], test_df['pred'])
            #     auc_for_sample_sizes[current_sample_size] = auc
                
            #     current_sample_size += step_size
              
            train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['sig_cancer'], random_state=random_state)
            test_df['pred'] = dataset.apply(get_baseline_predictions, axis=1)
            auc = roc_auc_score(test_df['sig_cancer'], test_df['pred'])
            sample_sizes = np.arange(initial_sample_size, len(dataset)*0.8, step_size)
            
            for current_sample_size in sample_sizes:
                auc_for_sample_sizes[current_sample_size] = auc
             
            auc_results_total[dataset_name].append(auc_for_sample_sizes)
                

    # Averaging results and creating a DataFrame for plotting
    average_results = {}
    for dataset_name, auc_list in auc_results_total.items():
        auc_df = pd.DataFrame(auc_list).mean(axis=0)  
        average_results[dataset_name] = auc_df

    # Compute overall averages across all datasets
    overall_averages = pd.DataFrame(average_results).mean(axis=1)
    overall_std = pd.DataFrame(average_results).std(axis=1)

    # Combine the dataset-specific averages with the overall average
    average_results['average'] = pd.concat([overall_averages, overall_std], axis=1)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsampling/baseline'):
        os.makedirs(directory + '/subsampling/baseline')

    for dataset_name,value in average_results.items():
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_samples', 'auc', 'std']
        else:
            value.columns=['num_samples', 'auc']
        value.to_csv(f"{directory}/subsampling/baseline/baseline_{dataset_name}.csv", index=False)
        
    return average_results
    
def main():
    # Parameters
    initial_sample_size = 10  # Starting sample size
    step_size = 20           # Increment size for each iteration
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    silos = apply_spline_transformation(silos)

    for a in ['Turkey', 'Finland']:
        silos.pop(a)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_baseline = run_baseline_sub_sampling(silos, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed)

    # Print the AUC matrix. For demonstration only
    print("\n### AUC Matrix for subsampling with baseline training ###")
    average_results_baseline.pop('average')
    print(pd.DataFrame(average_results_baseline, columns=client_names))

if __name__ == "__main__":
    main()