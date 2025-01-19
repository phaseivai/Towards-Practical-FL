import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from utils import apply_spline_transformation
from tqdm import tqdm
import os

from collections import defaultdict

'''This file trains local models with subsampling'''

def run_local_sub_sampling(datasets, n_repeats= 1, initial_sample_size = 50, 
                           step_size = 25, seed = 1234, directory = 'results'):
    
    np.random.seed(seed)

    auc_results_total = defaultdict(list) # Stores {dataset_name: [{sample_size: auc}, ...]}

    for repeat in tqdm(range(n_repeats)):
        random_state = np.random.randint(0, 10000000)

        for dataset_name, dataset in datasets.items():
            labels = dataset.iloc[:, 0]
            features = dataset.iloc[:, 1:]

            # Perform stratified sampling to get test split
            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            for train_idx, test_idx in stratified_split.split(features, labels):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            auc_results_local = {}

            current_sample_size = initial_sample_size
            # max_sample_size = min(len(X_train), 600)
            max_sample_size = len(X_train)

            while current_sample_size <= max_sample_size:
                # Perform stratified sampling for the current sample size, to get train split
                sample_split = StratifiedShuffleSplit(n_splits=1, train_size=current_sample_size, random_state=random_state)
                for sample_train_idx, _ in sample_split.split(X_train, y_train):
                    X_sampled_train = X_train.iloc[sample_train_idx]
                    y_sampled_train = y_train.iloc[sample_train_idx]

                model = LogisticRegression(solver='lbfgs', max_iter=10000)
                model.fit(X_sampled_train, y_sampled_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)

                auc_results_local[current_sample_size] = auc

                current_sample_size += step_size

            auc_results_total[dataset_name].append(auc_results_local)

    average_results = {}

    for dataset_name, auc_list in auc_results_total.items():
        auc_df = pd.DataFrame(auc_list).mean(axis=0)  
        average_results[dataset_name] = auc_df

    overall_averages = pd.DataFrame(average_results).mean(axis=1)
    overall_std = pd.DataFrame(average_results).std(axis=1)

    average_results['average'] = pd.concat([overall_averages, overall_std], axis=1)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsampling/local'):
        os.makedirs(directory + '/subsampling/local')

    for dataset_name,value in average_results.items():
        # value = pd.DataFrame(value)
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_samples', 'auc', 'std']
        else:
            value.columns=['num_samples', 'auc']
        value.to_csv(f"{directory}/subsampling/local/local_{dataset_name}.csv", index=False)
        
    return average_results


def main():
    # Parameters
    initial_sample_size = 50  # Starting sample size
    step_size = 25            # Increment size for each iteration
    n_repeats = 1             # Number of Monte-Carlo simulation rounds
    seed = 1234               # Fixed seed for reproducibility
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    silos = apply_spline_transformation(silos)

    for a in ['Turkey', 'Finland']:
        silos.pop(a)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_local = run_local_sub_sampling(silos, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed)
    
    # Print the AUC matrix for demonstration only
    print("\n### AUC Matrix for subsampling whe local training ###")
    average_results_local.pop('average')
    print(pd.DataFrame(average_results_local, columns=client_names))

if __name__ == "__main__":
    main()