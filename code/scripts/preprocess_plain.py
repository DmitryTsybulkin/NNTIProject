# %%
import multiprocessing as mp
import os

import numpy as np
from datasets import (
    load_dataset,
    Audio,
    load_from_disk
)
from huggingface_hub import login
from transformers import (
    AutoFeatureExtractor
)

# %%
# login to Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=HF_TOKEN)

print("Logged in to Hugging Face successfully!")

# %%
# model_id = "facebook/mms-300m"
model_id = "utter-project/mHuBERT-147"
# model_id = "facebook/wav2vec2-xls-r-300m"

max_duration = 5  # in seconds

# %%
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True,
)

# %%
if os.path.exists('./nnti-dataset'):
    print("Loading dataset from disk cache…")
    dataset = load_from_disk('./nnti-dataset')
else:
    print("Loading dataset from Hugging Face Hub…")
    dataset = load_dataset('badrex/nnti-dataset-full')
    dataset.save_to_disk('./nnti-dataset')

# %%
# check the strucutre of the dataset object
print(f"dataset['train']: {dataset['train']}")

# %%
# check the strucutre of one training sample (before decoding)
print(f"dataset['train'][0]: {dataset['train'][0]}")

# %%
# shuffle the dataset
train_ds = dataset['train'].shuffle(seed=42)
valid_ds = dataset['validation'].shuffle(seed=42)

# resample to 16kHz
train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

# %%
# based on the model typel, set input features key
if model_id == "facebook/w2v-bert-2.0":
    input_features_key = "input_features"
else:
    input_features_key = "input_values"

# %%
# get the set of languages
LABELS = train_ds.unique('language')

sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages: {sorted_labels}")

str_to_int = { s: i for i, s in enumerate(LABELS) }

ALL_SPEAKERS = sorted(set(train_ds['speaker_id']) | set(valid_ds['speaker_id']))
spk_to_int   = {s: i for i, s in enumerate(ALL_SPEAKERS)}
num_speakers = len(ALL_SPEAKERS)
print(f"Total speakers: {num_speakers}")

keep_cols = ['speaker_id', 'language']

# %%
def preprocess_function(examples):
    arrays = [x["array"] for x in examples["audio_filepath"]]
    inputs = feature_extractor(
        arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    inputs["spk_label"] = [spk_to_int[x] for x in examples["speaker_id"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    inputs["length"] = [len(f) for f in inputs[input_features_key]]
    return inputs


AVAILABLE_CPU = mp.cpu_count()
cache_base = f"./processed_data/{model_id.split('/')[-1]}_{max_duration}sec"

def load_or_create(ds, split_name: str):
    path = f"{cache_base}/{split_name}"
    if os.path.exists(path):
        print(f"  Loading cached {split_name} from {path}")
        return load_from_disk(path)
    print(f"  Preprocessing {split_name}…")
    result = ds.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        num_proc=AVAILABLE_CPU,
        remove_columns=[c for c in ds.column_names if c not in keep_cols],
    )
    result.save_to_disk(path)
    return result


train_ds_encoded = load_or_create(train_ds, 'train')
valid_ds_encoded = load_or_create(valid_ds, 'valid')
print(f"Train size: {len(train_ds_encoded)} | Valid size: {len(valid_ds_encoded)}")