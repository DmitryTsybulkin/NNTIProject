import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# --- 1. LOAD DATA ---
print("Loading files...")
data = np.load("bias_correct_embeddings2.npy")  # shape: (8689, 768)
df = pd.read_csv("bias_correct_metadata2.csv")


# This takes 50 samples from EACH language
samples_per_language = 400

df_subset = df.groupby('language').apply(
    lambda x: x.sample(n=min(len(x), samples_per_language), random_state=42)
).reset_index(drop=True)

# Align embeddings
subset_indices = df_subset.index.values # Note: reset_index might change this, 
                                        # better to capture indices BEFORE reset_index
# Correct way to get original indices:
original_indices = df.groupby('language').apply(
    lambda x: x.sample(n=min(len(x), samples_per_language), random_state=42)
).index.get_level_values(1)

data_subset = data[original_indices]
df_subset = df.loc[original_indices]

# Safety check: ensure they match in size
assert len(df_subset) == data_subset.shape[0], f"Error: Metadata has {len(df_subset)} rows but embeddings has {data_subset.shape[0]}!"


# --- 2. CLEAN & NORMALIZE ---
# Fixes the 'matmul' overflow/divide-by-zero errors
data_subset = np.nan_to_num(data_subset)
data_subset = StandardScaler().fit_transform(data_subset)

# --- 3. COMPUTE T-SNE ---
print("Computing t-SNE for 8,689 samples. This usually takes 1-2 minutes...")
# init='pca' makes the plot more stable/reproducible
tsne = TSNE(n_components=2, perplexity=40, init='pca', random_state=42, verbose=1)
proj_2d = tsne.fit_transform(data_subset)

# Save coordinates to our dataframe
df_subset['x'] = proj_2d[:, 0]
df_subset['y'] = proj_2d[:, 1]

# --- 4. PLOT CONFIGURATION ---
plt.figure(figsize=(14, 9))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


#language_markers = {'santali': 'o', 'dogri': 's'}
# Create the scatter plot
# We use 'husl' because it provides 22 distinct colors better than 'tab10'
scatter = sns.scatterplot(
    data=df_subset,
    x='x', y='y',
    hue='language',
    palette='husl',
    alpha=0.6,
    s=25,
    edgecolor='none',
    legend='brief'
)

# --- 5. LEGEND & LABELS ---
# Since speaker_id has too many entries, we only want to show the Language Shapes
handles, labels = scatter.get_legend_handles_labels()

# Find where the 'language' section starts in the legend handles
# Usually, Seaborn lists the Hue labels first, then the Style labels
try:
    lang_start = labels.index('language')
    plt.legend(handles[lang_start:], labels[lang_start:], 
               title="Language Shapes", bbox_to_anchor=(1.02, 1), loc='upper left')
except ValueError:
    plt.legend(title="Language & Speaker", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.title("mHuBERT: Language (Shape) vs. Speaker (Color) Distribution", pad=20)
plt.tight_layout()
plt.savefig("speaker_hue_language_shape.png", dpi=300)
plt.show()