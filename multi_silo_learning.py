import numpy as np
import pickle
from utils import apply_spline_transformation
from centralized import run_centralized_training
from fl import run_federated_training
from lso import run_lso_training
import pandas as pd
import argparse
import os

'''This file runs the centralized, federated, and LSO training for the matrix in Figure 4'''

def calc_color(color):
    return 100*(color-50)/(100-50)


# Function to generate LaTeX matrix code with row and column labels
def generate_latex_matrix(auc_matrix, dataset_names, client_pseud, dataset_sizes):
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

        latex_code += "    " + row_latex
    
    latex_code += "  };\n"  # Close the matrix

    # Add labels for the two extra rows: Centralized and Federated training
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-1-1.west) {{CEN}};\n"
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-2-1.west) {{FL}};\n"
    
    # Add row labels using the custom node format
    for i, dataset_name in enumerate(client_pseud):
        latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-{i+3}-1.west) {{$\\overline{{\\text{{{dataset_name}}}}}$}};\n"
        
    
    # Add column labels using the custom node format
    for j, dataset_name in enumerate(client_pseud):
        latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{j+1}.north) {{{dataset_name}}};\n"
    
    # Add labels for the two extra columns: Average and Weighted Average
    latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{len(dataset_names)+1}.north) {{AV-H}};\n"
    latex_code += f"  \\node[anchor=south] at ([yshift=+0.16cm]m-1-{len(dataset_names)+2}.north) {{AV-W}};\n"
    
    diag = 100*np.mean([auc_matrix[i+2][i] for i in range(num_clients)])
    
    latex_code += f"\n"
    latex_code += f"\\node[name=diag11, fill=orange!80!red!{calc_color(diag):.1f}!cyan!80!white] {{{diag:.1f}}}; & \\\\\n"

    
    # latex_code += "\\end{tikzpicture}"
    
    return latex_code


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
seed = 1234               # Fixed seed for reproducibility
n_repeats = args.monte_carlo        # Monte Carlo repetitions
num_rounds = args.federated_rounds      #Number of Federated Rounds

with open(args.data+'/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
silos = apply_spline_transformation(silos)

for a in ['Turkey', 'Finland']:
    silos.pop(a)
    
num_clients = len(silos)
client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
client_pseud = ['es', 'kr', 'de1', 'ch2', 'ch1', 'us2', 'us1', 'nl', 'it', 'de2', 'uk']

print("\n Running centralized training...")    
final_auc_matrix_cl = run_centralized_training(silos, client_names,n_repeats=n_repeats, seed=seed)
print("\n Running federated training...")
final_auc_matrix_fl = run_federated_training(silos, client_names,n_repeats=n_repeats, seed=seed)
print("\n Running leave silo out training...")
final_auc_matrix_lso = run_lso_training(silos, client_names,n_repeats=n_repeats, seed=seed)
print("\n Finished")

# Stack the full matrix
final_auc_matrix = np.vstack([final_auc_matrix_cl.reshape(1, -1), final_auc_matrix_fl.reshape(1, -1), final_auc_matrix_lso])

# Generate the LaTeX code for the AUC matrix with dataset labels
latex_auc_matrix = generate_latex_matrix(final_auc_matrix, client_names, client_pseud, dataset_sizes = [len(silos[key]) for key in client_names])

f = open( args.output + "/matrix_msl.txt", "w")
f.write(latex_auc_matrix)
f.close()

pd.DataFrame(final_auc_matrix).to_csv(args.output + '/msl_matrix.csv', index=False, header=False)