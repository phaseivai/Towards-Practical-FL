from utils import apply_spline_transformation
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from utils import Client, Server, CustomLogisticRegression
from tqdm import tqdm
import os

'''This file trains fl models with subsampling'''

def run_fl_sub_sampling(silos, num_rounds=1, n_repeats=1, initial_sample_size=50,
                                                   step_size=25, seed=1234, directory = 'results'):

    np.random.seed(seed)

    auc_results_total = defaultdict(list)  # Stores {dataset_name: [{sample_size: auc}, ...]}

    for repeat in tqdm(range(n_repeats)):
        # Initialize new random state for each repeat
        random_state = np.random.randint(0, 10000000)

        server = Server()
        clients = []
        
        for client_name, client_data in silos.items():
            client = Client(df=client_data, name=client_name, random_state=random_state)
            clients.append(client)
        
        # Process each dataset
        for dataset_name, dataset in silos.items():
            current_sample_size = initial_sample_size
            max_sample_size = int(len(dataset) * 0.8)  # Max training size (80% of dataset)
            
            auc_for_sample_sizes = {}  # {sample_size: auc} for this repetition
            
            # max_sample_size = min(max_sample_size, len(dataset))
            while current_sample_size <= max_sample_size:
                # For each sample size, perform federated training and aggregation
                for round_num in range(num_rounds):
                    client_updates = []
                    for client in clients:
                        if client.name == dataset_name:
                            # Train on the subsampled dataset for current sample size
                            coef, intercept, num_samples = client.train(initial_sample_size=current_sample_size)
                        else:
                            # Train on the full dataset for all other clients
                            coef, intercept, num_samples = client.train()

                        if coef is not None and intercept is not None:
                            client_updates.append((coef, intercept, num_samples))

                    # Aggregate client updates to produce global parameters
                    global_coef, global_intercept = server.aggregate(client_updates)

                    # Send global parameters back to each client
                    for client in clients:
                        client.set_params(global_coef, global_intercept)
                
                for client in clients:
                    if client.name == dataset_name:
                        _, _, auc = client.evaluate(use_full_dataset=False)  # Evaluate on test split
                        auc_for_sample_sizes[current_sample_size] = auc

                current_sample_size += step_size  # Increment sample size for next iteration

            auc_results_total[dataset_name].append(auc_for_sample_sizes)

    # Averaging results and creating a DataFrame for plotting
    average_results = {}
    for dataset_name, auc_list in auc_results_total.items():
        # Each element in auc_list is a dict of {sample_size: auc_score}
        auc_df = pd.DataFrame(auc_list).mean(axis=0)
        average_results[dataset_name] = auc_df
        
    overall_averages = pd.DataFrame(average_results).mean(axis=1)
    overall_std = pd.DataFrame(average_results).std(axis=1)

    average_results['average'] = pd.concat([overall_averages, overall_std], axis=1)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsampling/fl'):
        os.makedirs(directory + '/subsampling/fl')

    for dataset_name,value in average_results.items():
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_samples', 'auc', 'std']
        else:
            value.columns=['num_samples', 'auc']
        value.to_csv(f"{directory}/subsampling/fl/fl_{dataset_name}.csv", index=False)
        
    return average_results

def main():
    # Parameters
    initial_sample_size = 50  # Starting sample size
    step_size = 25           # Increment size for each iteration
    seed = 1234               # Fixed seed for reproducibility
    num_rounds = 1
    n_repeats = 1
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    silos = apply_spline_transformation(silos)

    for a in ['Turkey', 'Finland']:
        silos.pop(a)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_local = run_fl_sub_sampling(silos, num_rounds=num_rounds, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed)
    
    # Print the AUC matrix. For demonstration only
    print("\n### AUC Matrix for subsampling with federated learning training ###")
    average_results_local.pop('average')
    print(pd.DataFrame(average_results_local, columns=client_names))

if __name__ == "__main__":
    main()