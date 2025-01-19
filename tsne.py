import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

'''This file generates tSNE for the appendix'''

# dataset from pickle
with open('data/data.pkl', 'rb') as file:
    silos = pickle.load(file)
    
# create output directory
if not os.path.exists("results/tsne"):
        os.makedirs("results/tsne")


data_combined = pd.DataFrame() 
labels_combined = []  

for dataset_name, df in silos.items():
    if dataset_name == 'Turkey' or dataset_name == 'Finland':
        continue
    data_combined = pd.concat([data_combined, df.drop(['sig_cancer', '5ARI'], axis=1)])
    labels_combined += [dataset_name] * len(df)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data_combined)

tsne_df = pd.DataFrame(tsne_results, columns=['tSNE1', 'tSNE2'])
tsne_df['dataset'] = labels_combined 

for dataset_name in silos.keys():
    if dataset_name == 'Turkey' or dataset_name == 'Finland':
        continue
    tsne_df[tsne_df['dataset'] == dataset_name].to_csv(f"results/tsne/tsne_{dataset_name}.csv")

# The actual plot for the paper is done with latex tikz. This is for demonstrative purposes only
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tSNE1', y='tSNE2', hue='dataset', data=tsne_df, palette='tab20')
plt.title('t-SNE on Train Hospitals')
plt.show()