from sklearn.model_selection import train_test_split
from utils import apply_spline_transformation
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from utils import Client, Server
from tqdm import tqdm
import os

'''This file runs LSO training with subsampling'''

def run_lso_sub_sampling(silos, client_names, num_rounds=1, n_repeats=1, initial_sample_size=50,
                                                   step_size=25, seed=1234, directory = 'results'):

    np.random.seed(seed)

    auc_results_total = defaultdict(list)  # Stores {dataset_name: [{sample_size: auc}, ...]}

    for repeat in tqdm(range(n_repeats)):
        random_state = np.random.randint(0, 10000000)
        
        server = Server()
        
        # Process each dataset
        for dataset_name, dataset in silos.items():
            
            participating_clients = [
                    Client(df=silos[name], name=name, random_state=random_state) 
                    for name in client_names 
                    if dataset_name != name
                ]
            
            for round_num in range(num_rounds):
                    client_updates = []
                    for client in participating_clients:
                        coef, intercept, num_samples = client.train()

                        if coef is not None and intercept is not None:
                            client_updates.append((coef, intercept, num_samples))

                    # Aggregate client updates to produce global parameters
                    global_coef, global_intercept = server.aggregate(client_updates)

                    # Send global parameters back to each client
                    for client in participating_clients:
                        client.set_params(global_coef, global_intercept)
                    
            # Adopt the global model for excluded client
            excluded_client = Client(df=dataset, name=dataset_name, random_state=random_state)
            excluded_client.set_params(global_coef, global_intercept)
            
            current_sample_size = initial_sample_size
            max_sample_size = len(dataset)
            # max_sample_size = 600
            
            auc_for_sample_sizes = {}  # {sample_size: auc} for this repetition
            
            # max_sample_size = min(len(dataset), 600)
            # while current_sample_size <= max_sample_size:
            #     test_fraction = current_sample_size / max_sample_size

            #     X = dataset.iloc[:, 1:].values
            #     y = dataset.iloc[:, 0].values

            #     if test_fraction < 1:
            #         X_train, X_test, y_train, y_test = train_test_split(
            #             X, y, test_size=test_fraction, random_state=random_state, stratify=y
            #         ) 
            #     else:
            #         X_test = X
            #         y_test = y
                
            #     # Set the sliced test split for excluded client
            #     excluded_client.X_test = X_test
            #     excluded_client.y_test = y_test
            #     _, _, auc = excluded_client.evaluate(use_full_dataset=False)  # Evaluate on test split
            #     auc_for_sample_sizes[current_sample_size] = auc  # Store AUC for this sample size

            #     current_sample_size += step_size  

            
            X = dataset.iloc[:, 1:].values
            y = dataset.iloc[:, 0].values
            
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
            excluded_client.X_test = X_test
            excluded_client.y_test = y_test
            _, _, auc = excluded_client.evaluate(use_full_dataset=False)  # Evaluate on test split
            
            sample_sizes = np.arange(initial_sample_size, len(dataset)*0.8, step_size)
            for current_sample_size in sample_sizes:
                auc_for_sample_sizes[current_sample_size] = auc
                
            # Append the result of this repetition for this dataset
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
    
    if not os.path.exists(directory + '/subsampling/lso'):
        os.makedirs(directory + '/subsampling/lso')

    for dataset_name,value in average_results.items():
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_samples', 'auc', 'std']
        else:
            value.columns=['num_samples', 'auc']
        value.to_csv(f"{directory}/subsampling/lso/lso_{dataset_name}.csv", index=False)
    
    return average_results

def main():
    # Parameters
    initial_sample_size = 10  # Starting sample size
    step_size = 20           # Increment size for each iteration
    seed = 1234               # Fixed seed for reproducibility
    num_rounds = 1
    n_repeats = 1           # Number of Monte Carlo rounds
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    silos = apply_spline_transformation(silos)

    for a in ['Turkey', 'Finland']:
        silos.pop(a)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_local = run_lso_sub_sampling(silos, client_names, num_rounds=num_rounds, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed)
    
    # Print the AUC matrix. For demonstration only
    print("\n### AUC Matrix for subsampling with leave silo out training ###")
    average_results_local.pop('average')
    print(pd.DataFrame(average_results_local, columns=client_names))

if __name__ == "__main__":
    main()