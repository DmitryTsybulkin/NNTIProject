import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# --- 1. LOAD DATA ---
print("Loading files...")
data = np.load("bias_correct_embeddings.npy")  # shape: (8689, 768)
df = pd.read_csv("bias_correct_metadata.csv")

# --- 2. FILTER FOR SPECIFIC LANGUAGES ---
target_languages = ['punjabi','dogri','malayalam','kannada']
#target_languages = ['urdu', 'hindi']
df_filtered = df[df['language'].isin(target_languages)].copy()
data_filtered = data[df_filtered.index.values]

"""
# This takes 50 samples from EACH language
samples_per_language = 100

df_subset = df.groupby('language').apply(
    lambda x: x.sample(n=min(len(x), samples_per_language), random_state=42)
).reset_index(drop=True)

# Align embeddings
subset_indices = df_subset.index.values # Note: reset_index might change this, 
                                        # better to capture indices BEFORE reset_index
# Get original indices:
original_indices = df.groupby('language').apply(
    lambda x: x.sample(n=min(len(x), samples_per_language), random_state=42)
).index.get_level_values(1)

data_subset = data[original_indices]
df_subset = df.loc[original_indices]

# Safety check: ensure they match in size
assert len(df_subset) == data_subset.shape[0], f"Error: Metadata has {len(df_subset)} rows but embeddings has {data_subset.shape[0]}!"
"""

# Clean and Normalize

data_subset = np.nan_to_num(data_filtered)
data_subset = StandardScaler().fit_transform(data_subset)

# Compute T-SNE 
print("Computing t-SNE for 8,689 samples...")
# init='pca' makes the plot more stable/reproducible
tsne = TSNE(n_components=2, perplexity=40, init='pca', random_state=42, verbose=1)
proj_2d = tsne.fit_transform(data_subset)

# Save coordinates to our dataframe
df_filtered['x'] = proj_2d[:, 0]
df_filtered['y'] = proj_2d[:, 1]

# Plot 
plt.figure(figsize=(14, 9))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

#language_markers = {'hindi': 'o', 'urdu': 's'}

# Create the scatter plot
scatter = sns.scatterplot(
    data=df_filtered,
    x='x', y='y',
    hue='language',
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
    plt.legend(title="Language & Scenario", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.title("mHuBERT: Language (Shape) vs. Scenario (Color) Distribution", pad=20)
plt.tight_layout()
plt.savefig("scenario_hue_language_shape.png", dpi=300)
plt.show()