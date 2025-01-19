import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

'''This file calculates the stats for features of the datasets, as depicted in Figure 3 '''

display_in_python = True

def calculate_global_stats(datasets, parameter):
    '''This function calculates global max, min and mean'''    
    combined_age_series = pd.concat([df[parameter] for key,df in datasets.items() if key != 'Turkey' and key != 'Finland'])
    return combined_age_series.min(), combined_age_series.max(), combined_age_series.mean()

def create_histogram(df, parameter, bins):
    # Use np.histogram with density=True to normalize the histogram
    hist_values, bin_edges = np.histogram(df[parameter], bins=bins, density=True)
    return hist_values, bin_edges

def calculate_placement(value, bin_edges, bin_midpoints, offset=0):
    '''This function plots the global mean within the histogram distribution'''
    placement = (value - bin_midpoints[np.digitize(value, bin_edges)-1])/(bin_midpoints[np.digitize(value, bin_edges)] \
        - bin_midpoints[np.digitize(value, bin_edges)-1]) + np.digitize(value, bin_edges) + offset
    return placement

# dataset from pickle
with open('data/data.pkl', 'rb') as file:
    silos = pickle.load(file)
    
    
directories = ['results/features/features_age', 'results/features/features_psa', 
               'results/features/features_pirads']

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Age

# Calculate global min, max, and mean age
global_min_age, global_max_age, global_mean_age = calculate_global_stats(silos, 'age')

# number of bins for the age histogram 
num_bins_age = 7
bins_age = np.linspace(global_min_age, global_max_age, num=num_bins_age + 1)

bin_age = {}

for key, df in silos.items():
    if key != 'Turkey' and key != 'Finland':
        hist_values, bin_edges = create_histogram(df, 'age', bins_age)
        
        bin_age[key] = {
            'hist_values': hist_values,
            'bin_edges': bin_edges
        }
        
        bin_indices = np.arange(1, num_bins_age + 1)
        bin_pairs = pd.DataFrame({
            'bin_index': bin_indices,
            'bin_value': hist_values
        })

        bin_pairs.to_csv(f'results/features/features_age/{key}.csv', index=False)

# These are the plot coordinates for latex
bin_midpoints = (bins_age[:-1] + bins_age[1:])/2
plot_min_age = np.arange(1, num_bins_age + 1)[0] - 0.5
plot_max_age = np.arange(1, num_bins_age + 1)[-1] + 0.5
plot_mean_age = calculate_placement(global_mean_age, bins_age, bin_midpoints)
print(f"plot min age position = {plot_min_age}, plot min age value = {global_min_age} \n plot mean age position = {plot_mean_age}, \
    plot mean age value = {global_mean_age} \n plot max age position = {plot_max_age}, plot max age value = {global_max_age}")


# The actual plot for the paper is done with latex tikz. This is for demonstrative purposes only
if display_in_python:
    for key, result in bin_age.items():
        bin_midpoints = (result['bin_edges'][:-1] + result['bin_edges'][1:]) / 2
        plt.bar(bin_midpoints, result['hist_values'], width=np.diff(result['bin_edges']), alpha=0.5, label=key)

    # # Vertical lines for global min, global max, and global mean
    # plt.axvline(global_min_age, color='red', linestyle='--', linewidth=2, label='Global Min')
    # plt.axvline(global_max_age, color='green', linestyle='--', linewidth=2, label='Global Max')
    # plt.axvline(global_mean_age, color='blue', linestyle='--', linewidth=2, label='Global Mean')

    plt.title(f'Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
# PSA

global_min_psa, global_max_psa, global_mean_psa = calculate_global_stats(silos, 'PSA')

# The bins are space linearly except for the last one, which contains extreme values of PSA
first_interval_bins = 6
# construct the fisrt bins
number_of_means = 2
first_part_bins = np.linspace(global_min_psa, number_of_means * global_mean_psa, first_interval_bins + 1)
bin_width = first_part_bins[1] - first_part_bins[0]  

# construct the last bin
last_bin_edge = global_max_psa
bins_psa = np.concatenate([first_part_bins, [last_bin_edge]])

# The first 6 bins will be plotted normally
first_bin_edges = np.linspace(global_min_psa, number_of_means * global_mean_psa, first_interval_bins + 1)

# For the plot, the last bin edge is artificially kept the same size as the others
plot_bin_edges = np.concatenate([first_bin_edges, [first_bin_edges[-1] + bin_width]])
real_bin_edges = np.append(first_bin_edges, global_max_psa)


psa_shift = 9
bin_psa = {}

for key, df in silos.items():
    if key != 'Turkey' and key != 'Finland':
        hist_values, bin_edges = create_histogram(df, 'PSA', bins_psa)

        bin_indices = np.arange(1 + psa_shift, len(bins_psa) + psa_shift)
        
        # Create DataFrame for the current dataset
        bin_pairs = pd.DataFrame({
            'bin_index': np.append(psa_shift + 0.5, bin_indices),
            'bin_value': np.append(0, hist_values)
        })

        bin_psa[key] = bin_pairs
        
        bin_pairs.to_csv(f'results/features/features_psa/{key}.csv', index=False)
    
# These are the plot coordinates for latex
bin_midpoints = (bins_psa[:-1] + bins_psa[1:])/2
plot_min_psa = np.arange(1 + psa_shift, len(bins_psa) + psa_shift)[0] - 0.5
plot_max_psa = np.arange(1 + psa_shift, len(bins_psa) + psa_shift)[-1] + 0.5
plot_mean_psa = calculate_placement(global_mean_psa, real_bin_edges, bin_midpoints, psa_shift)
print(f"plot min psa position = {plot_min_psa}, plot min psa value = {global_min_psa} \n plot mean psa position = {plot_mean_psa}, \
    plot mean psa value = {global_mean_psa} \n plot max psa position = {plot_max_psa}, plot max psa value = {global_max_psa}")

# The actual plot for the paper is done with latex tikz. This is for demonstrative purposes only
if display_in_python:
    plt.figure(figsize=(8, 6))
    for key, result in bin_psa.items():
        plt.bar([a for a in range(len(bin_midpoints))], result['bin_value'][1:],width=1, alpha=0.5, label=key)
        
    plt.title(f'PSA Distribution')
    plt.xlabel('PSA')
    plt.ylabel('Density')
    plt.xticks([a for a in range(len(bin_midpoints))], np.round(bin_midpoints,1))

    plt.tight_layout()
    plt.legend()
    plt.show()
    
    
# PIRADS
pirads_shift = 18
pirads_coeff = 0.15


bin_pirads = {}
for key,df in silos.items():
    if key != 'Turkey' and key != 'Finland':
        value_counts = df['PIRADS'].value_counts()
        value_counts = value_counts.reset_index()
        value_counts.columns = ['bin_index', 'bin_value'] 
        real_pirads = value_counts['bin_index']
        value_counts['bin_index'] = value_counts['bin_index'] + pirads_shift
        value_counts['bin_value'] = (value_counts['bin_value'] / value_counts['bin_value'].sum()) * pirads_coeff
        bin_pirads[key] = value_counts
        value_counts.to_csv(f'results/features/features_pirads/{key}.csv', index=False)
        
for pirads, position in zip(real_pirads, value_counts['bin_index']):
        print(f"The PIRADS {pirads} is plotted at {position}")
    

# The actual plot for the paper is done with latex tikz. This is for demonstrative purposes only
if display_in_python:
    plt.figure(figsize=(8, 6))
    for key, result in bin_pirads.items():
        plt.bar(result['bin_index']-pirads_shift, result['bin_value'], width=1, alpha=0.5, label=key)
        
    plt.title(f'PIRADS Distribution')
    plt.xlabel('PIRADS')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.legend()
    plt.show()