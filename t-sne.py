import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load data 

data = np.load("bias_correct_embeddings.npy")  
df = pd.read_csv("bias_correct_metadata.csv") 

# Filter to only 4 languages for better visualization
target_languages = ['punjabi','dogri','malayalam','kannada'] 
#target_languages = ['urdu', 'hindi']
df_filtered = df[df['language'].isin(target_languages)].copy() # Take subset of data which contains our target languages.
data_filtered = data[df_filtered.index.values] # Align the filtered metadata with the corresponding embeddings.


# Clean and Normalize

data_subset = np.nan_to_num(data_filtered) # Fixes overflow/divide-by-zero errors
data_subset = StandardScaler().fit_transform(data_subset) # Standardize features to have mean=0 and variance=1, which helps t-SNE perform better.

# Compute t-SNE 

# init='pca' makes the plot more stable/reproducible
tsne = TSNE(n_components=2, perplexity=40, init='pca', random_state=42, verbose=1) # Perplexity is a key hyperparameter that balances local vs global structure in the data. 40 is a common choice for datasets of this size.
proj_2d = tsne.fit_transform(data_subset) # Project the high-dimensional data down to 2D for visualization

# Save coordinates to our dataframe

df_filtered['x'] = proj_2d[:, 0] 
df_filtered['y'] = proj_2d[:, 1]

# Plot 

plt.figure(figsize=(14, 9))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

#language_markers = {'hindi': 'o', 'urdu': 's'} # Different shapes for different languages

# Create the scatter plot
scatter = sns.scatterplot(
    data=df_filtered,
    x='x', y='y',
    hue='language', # Color points by language
    palette='husl', # We use 'husl' because it provides 22 distinct colors better than 'tab10'
    alpha=0.6,
    s=100,
    edgecolor='none',
    legend='brief'
)

# Legend configuration

# Since speaker_id has too many entries, we only want to show the language shapes
handles, labels = scatter.get_legend_handles_labels()

# Find where the language section starts in the legend handles
try:
    lang_start = labels.index('language')
    plt.legend(handles[lang_start:], labels[lang_start:], 
               title="Language Shapes", bbox_to_anchor=(1.02, 1), loc='upper left')
except ValueError:
    plt.legend(title="Language Clusters", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.title("mHuBERT: Language Clusters: Kannada, Punjabi, Malayalam, Dogri", pad=20)
plt.tight_layout()
plt.savefig("t-sne-targeted.png", dpi=300)
plt.show()