import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split  # for train/test split
from utils import apply_spline_transformation
from tqdm import tqdm
import pandas as pd
import argparse
import os

'''This file trains local models for the matrix in Figure 4'''

def calc_color(color):
    return 100*(color-50)/(100-50)


# Function to generate LaTeX matrix code with row and column labels
def generate_latex_matrix(auc_matrix, dataset_names, dataset_sizes):
    latex_code = ''
    # Start the LaTeX code for the matrix
    latex_code = "\\matrix[matrix of nodes, nodes={minimum size=1.0cm, anchor=center, font=\\large, text centered}, column sep=0.05cm, row sep=0.05cm] (m) {\n"
    
    # Add AUC values row by row with two additional columns (Average and Weighted Average)
    for i, row in enumerate(auc_matrix):
        avg_auc = np.mean(row)  # Average AUC
        weighted_avg_auc = np.average(row, weights=dataset_sizes)  # Weighted average AUC

        # Prepare each row's values for each AUC
        row_latex = ""
        for j, auc in enumerate(row):
            # Scale AUC to 50-100 for color
            auc_scaled = auc * 100
            
            # Construct the node with custom coloring
            row_latex += f"\\node[name=m-{i+1}-{j+1}, fill=orange!80!red!{calc_color(auc_scaled):.1f}!cyan!80!white] {{{auc_scaled:.1f}}}; & "
        
        # Add Average and Weighted Average as the last two columns
        row_latex += f"\\node[name=m-{i+1}-{len(row)+1}, fill=orange!80!red!{calc_color(avg_auc*100):.1f}!cyan!80!white] {{{avg_auc*100:.1f}}}; & "
        row_latex += f"\\node[name=m-{i+1}-{len(row)+2}, fill=orange!80!red!{calc_color(weighted_avg_auc*100):.1f}!cyan!80!white] {{{weighted_avg_auc*100:.1f}}}; \\\\\n"

        # Append this row to the final LaTeX code
        latex_code += "    " + row_latex
    
    latex_code += "  };\n"  # Close the matrix

    # Add row labels using the custom node format
    for i, dataset_name in enumerate(dataset_pseud):
        latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-{i+1}-1.west) {{{dataset_name}}};\n"
    
    # Add column labels using the custom node format
    for j, dataset_name in enumerate(dataset_pseud):
        latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{j+1}.north) {{{dataset_name}}};\n"
    
    # Add labels for the two extra columns: Average and Weighted Average
    latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{len(dataset_names)+1}.north) {{AV-H}};\n"
    latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{len(dataset_names)+2}.north) {{AV-W}};\n"
    
    diag = 100*np.mean([auc_matrix[i][i] for i in range(num_datasets)])
    
    latex_code += f"\n"
    latex_code += f"\\node[name=diag11, fill=orange!80!red!{calc_color(diag):.1f}!cyan!80!white] {{{diag:.1f}}}; & \\\\\n"

    
    # latex_code += "\\end{tikzpicture}"
    
    return latex_code

parser = argparse.ArgumentParser(description="A script with argparse defaults.")
parser.add_argument("--data", type=str, default="data", help="Data folder")
parser.add_argument("--output", type=str, default="results", help="Folder to store the output")
parser.add_argument("--monte_carlo", type=int, default=1, help="Number of Monte Carlo Simulations")
# Parse arguments
args = parser.parse_args()

# create output directory
if not os.path.exists(args.output):
        os.makedirs(args.output)
        
seed = 1234     
np.random.seed(seed)  # For reproducibility
n_repeats = args.monte_carlo     # Number of times to repeat the experiment

# dataset from pickle
with open(args.data + '/data.pkl', 'rb') as file:
    silos = pickle.load(file)

silos = apply_spline_transformation(silos)

for a in ['Turkey', 'Finland']:
    silos.pop(a)

# Number of datasets (m x m matrix)
num_datasets = len(silos)

# Dictionary of dataset names for referencing
dataset_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
dataset_pseud = ['es', 'kr', 'de1', 'ch2', 'ch1', 'us2', 'us1', 'nl', 'it', 'de2', 'gb']

# Initialize a list to hold all AUC matrices
auc_matrices = []

print("\n Running single silo training...")  

for repeat in tqdm(range(n_repeats)):
    
    random_state = np.random.randint(0, 10000000)
    
    # Create dictionaries for storing train/test features and labels
    test_features = {}
    test_labels = {}
    train_features = {}
    train_labels = {}
    
    # Initialize an m x m matrix for AUC values
    auc_matrix = np.zeros((num_datasets, num_datasets))
    
    for key, value in silos.items():
        labels = value.iloc[:, 0]  
        features = value.iloc[:, 1:] 

        features = features.values
        labels = labels.values
        
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=random_state)

        test_features[key] = X_test
        test_labels[key] = y_test

        train_features[key] = X_train
        train_labels[key] = y_train

    # Train models on each dataset and populate AUC matrix
    for i, key in enumerate(dataset_names):

        model = LogisticRegression(solver='lbfgs', max_iter=10000)
        model.fit(train_features[key], train_labels[key])

        # Test on each dataset
        for j, key1 in enumerate(dataset_names):
            
            y_pred_proba = model.predict_proba(test_features[key1])[:, 1]
            auc = roc_auc_score(test_labels[key1], y_pred_proba)
            
            # Store the AUC value in the matrix
            auc_matrix[i, j] = auc
        
    
    auc_matrices.append(auc_matrix)
    
    
print("\n Finished")
    
# Calculate the average AUC matrix across all repetitions
final_auc_matrix = np.mean(auc_matrices, axis=0)

# Generate the LaTeX code for the AUC matrix with dataset labels
latex_auc_matrix = generate_latex_matrix(final_auc_matrix, dataset_names, dataset_sizes = [len(silos[key]) for key in dataset_names])
f = open(args.output + "/matrix_ssl.txt", "w")
f.write(latex_auc_matrix)
f.close()

pd.DataFrame(final_auc_matrix).to_csv(args.output + '/ssl_matrix.csv', index=False, header=False)