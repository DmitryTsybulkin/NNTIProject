import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load data

data = np.load("bias_correct_embeddings2.npy") 
df = pd.read_csv("bias_correct_metadata2.csv")


# Number of samples per language for t-SNE 
samples_per_language = 400

# Samples requested number of samples per language and gets the indices for the subset of data.
original_indices = df.groupby('language').sample(
    n=samples_per_language, 
    random_state=42, 
    replace=False
).index

# Use those specific indices to allign dataframe and the embeddings
df_subset = df.loc[original_indices].reset_index(drop=True)
data_subset = data[original_indices] 


# Safety check: ensure they match in size
assert len(df_subset) == data_subset.shape[0], f"Error: Metadata has {len(df_subset)} rows but embeddings has {data_subset.shape[0]}!"


# Clean and Normalize

data_subset = np.nan_to_num(data_subset) # Fixes overflow/divide-by-zero errors
data_subset = StandardScaler().fit_transform(data_subset) # Standardize features to have mean=0 and variance=1, which helps t-SNE perform better.

# Compute t-SNE

# init='pca' makes the plot more stable/reproducible
tsne = TSNE(n_components=2, perplexity=40, init='pca', random_state=42, verbose=1) # Perplexity is a key hyperparameter that balances local vs global structure in the data. 40 is a common choice for datasets of this size.
proj_2d = tsne.fit_transform(data_subset) # Project the high-dimensional data down to 2D for visualization


# Save coordinates to our dataframe
df_subset['x'] = proj_2d[:, 0]
df_subset['y'] = proj_2d[:, 1]

# Plot
plt.figure(figsize=(14, 9))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


#language_markers = {'santali': 'o', 'dogri': 's'}
# Create the scatter plot

scatter = sns.scatterplot(
    data=df_subset,
    x='x', y='y',
    hue='language', # Color points by language
    palette='husl', # We use 'husl' because it provides 22 distinct colors better than 'tab10'
    alpha=0.6,
    s=25,
    edgecolor='none',
    legend='brief'
)

# Legend configuration

# Since speaker_id has too many entries, we only want to show the Language Shapes
handles, labels = scatter.get_legend_handles_labels()

# Find where the 'language' section starts in the legend handles
try:
    lang_start = labels.index('language')
    plt.legend(handles[lang_start:], labels[lang_start:], 
               title="Language Colors", bbox_to_anchor=(1.02, 1), loc='upper left')
except ValueError:
    plt.legend(title="Language Subset", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.title("mHuBERT: Language Subset", pad=20)
plt.tight_layout()
plt.savefig("t-sne-subset.png", dpi=300)
plt.show()