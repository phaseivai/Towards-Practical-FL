from subsample_fl import run_fl_sub_sampling
from subsample_lso import run_lso_sub_sampling
from subsample_local import run_local_sub_sampling
from subsampling_baseline import run_baseline_sub_sampling
import pickle
from utils import apply_spline_transformation
import argparse
import os
import pandas as pd

'''This file trains various models with subsampling to construct the Figure 5'''

parser = argparse.ArgumentParser(description="A script with argparse defaults.")
parser.add_argument("--data", type=str, default="data", help="Data folder")
parser.add_argument("--output", type=str, default="results", help="Folder to store the output")
parser.add_argument("--monte_carlo", type=int, default=1, help="Number of Monte Carlo Simulations")
parser.add_argument("--federated_rounds", type=int, default=1, help="Number of Federated Rounds")
# Parse arguments
args = parser.parse_args()

# create output directory
if not os.path.exists(args.output):
        os.makedirs(args.output)


 # Parameters
initial_sample_size = 10  # Starting sample size
step_size = 20           # Increment size for each iteration
seed = 1234               # Fixed seed for reproducibility
num_rounds = args.federated_rounds
n_repeats = args.monte_carlo

with open(args.data+'/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
silos = apply_spline_transformation(silos)

for a in ['Turkey', 'Finland']:
    silos.pop(a)
    
client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)

print("\n Running local subsampling...")
average_results_local = run_local_sub_sampling(silos, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed, directory = args.output)
print("\n Running federated learning subsampling...")
average_results_fl = run_fl_sub_sampling(silos, num_rounds=num_rounds, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed, directory = args.output)
print("\n Running leave-silo-out subsampling...")
average_results_lso = run_lso_sub_sampling(silos, client_names, num_rounds=num_rounds, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed, directory = args.output)

print("\n Running baseline subsampling...")
average_results_baseline = run_baseline_sub_sampling(silos, n_repeats=n_repeats, initial_sample_size=initial_sample_size,
                                                   step_size=step_size, seed=seed, directory = args.output)

print("\n Finished")

# Calculate the difference between local and FL for Figure 5
average_results_local['difference'] = average_results_local['average'] - average_results_fl['average']
average_results_local['difference'] = average_results_local['difference'].reset_index()
average_results_local['difference'].columns=['num_samples', 'auc_dif', 'std_dif']
average_results_local['difference'].to_csv(f"{args.output}/subsampling/local/local_difference.csv", index=False)

# Calculate the difference between LSO and FL for Figure 5
average_results_lso['average'].columns=['auc', 'std']
average_results_fl['average'].columns=['auc', 'std']
average_results_lso['average']['auc'] = average_results_lso['average'].iloc[0]['auc']

average_results_lso['difference']  = average_results_lso['average'] - average_results_fl['average']
average_results_lso['difference'] = average_results_lso['difference'].reset_index()
average_results_lso['difference'].columns = ['num_samples', 'auc_dif', 'std_dif']
average_results_lso['difference'].to_csv(f"{args.output}/subsampling/fl/fl_difference.csv")

# Calculate the difference between LSO and Local for Figure 5
local = pd.read_csv(f"{args.output}/subsampling/local/local_difference.csv")
fl = pd.read_csv(f"{args.output}/subsampling/fl/fl_difference.csv")
merged_df = pd.merge(local, fl, on='num_samples', suffixes=('_local', '_fl'))
lso = pd.DataFrame({
    'num_samples': merged_df['num_samples'],
    'auc_dif': merged_df['auc_dif_local'] - merged_df['auc_dif_fl']
})
lso.to_csv(f"{args.output}/subsampling/lso/lso_difference.csv")