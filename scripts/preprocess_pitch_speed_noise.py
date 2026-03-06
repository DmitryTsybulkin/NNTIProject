import multiprocessing as mp
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import Dataset
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from huggingface_hub import login
from transformers import (
    AutoFeatureExtractor,
)

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")
login(token=HF_TOKEN)
print("Logged in to Hugging Face!")

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
model_id                    = "utter-project/mHuBERT-147"
max_duration                = 5
sampling_rate               = 16000


# Augmentation settings
# pitch is pre-baked (slow GPU op, done once); speed+noise applied live in collator
PITCH_SEMITONES_RANGE = (-3.0, 3.0)   # semitones up/down
SPEED_RATE_RANGE      = (0.85, 1.15)  # speed factor range
NOISE_SNR_RANGE       = (15.0, 40.0)  # SNR dB range

input_features_key = "input_values"   # mHuBERT uses input_values

AVAILABLE_CPU = mp.cpu_count()

# GPU for pitch preprocessing (20-50x faster than CPU)
_AUG_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Augmentation device: {_AUG_DEVICE}")

# ──────────────────────────────────────────────
# FEATURE EXTRACTOR & DATA
# ──────────────────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

if os.path.exists('./nnti-dataset'):
    print("Loading dataset from disk cache…")
    dataset = load_from_disk('./nnti-dataset')
else:
    print("Loading dataset from Hugging Face Hub…")
    dataset = load_dataset('badrex/nnti-dataset-full')
    dataset.save_to_disk('./nnti-dataset')
print(f"dataset['train']: {dataset['train']}")
print(f"dataset['train'][0]: {dataset['train'][0]}")

train_ds = dataset['train'].shuffle(seed=42).cast_column(
    "audio_filepath", Audio(sampling_rate=sampling_rate))
valid_ds = dataset['validation'].shuffle(seed=42).cast_column(
    "audio_filepath", Audio(sampling_rate=sampling_rate))

LABELS        = train_ds.unique('language')
sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages: {sorted_labels}")
str_to_int    = {s: i for i, s in enumerate(LABELS)}
int_to_str    = {i: s for s, i in str_to_int.items()}
num_labels    = len(LABELS)

# ──────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
#
# SLOW (pitch)  → pre-baked into dataset cache on GPU, done once
# FAST (speed, noise) → applied live in collator, no GPU stall during training
# ──────────────────────────────────────────────

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).numpy()


def augment_pitch(arr: np.ndarray) -> np.ndarray:
    """GPU pitch shift — runs during preprocessing, cached to disk."""
    n_steps = int(random.uniform(*PITCH_SEMITONES_RANGE))
    with torch.no_grad():
        x       = _to_tensor(arr).to(_AUG_DEVICE)
        shifted = T.PitchShift(sample_rate=sampling_rate, n_steps=n_steps).to(_AUG_DEVICE)(x)
    return shifted.cpu().squeeze(0).numpy()


# Создаем кэш ресэмплеров ОДИН РАЗ в глобальной области видимости
SPEED_RATES = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
RESAMPLERS = {}
for rate in SPEED_RATES:
    new_sr = int(sampling_rate * rate)
    # T.Resample создается только один раз для каждой скорости
    RESAMPLERS[rate] = T.Resample(orig_freq=new_sr, new_freq=sampling_rate)

def augment_speed(arr: np.ndarray) -> np.ndarray:
    """Оптимизированный speed perturbation с использованием кэша."""
    rate = random.choice(SPEED_RATES) # Выбираем из готовых
    resampler = RESAMPLERS[rate]      # Берем готовый фильтр

    resampled = resampler(_to_tensor(arr))
    orig_len  = arr.shape[0]

    if resampled.shape[1] < orig_len:
        resampled = F.pad(resampled, (0, orig_len - resampled.shape[1]))
    else:
        resampled = resampled[:, :orig_len]
    return _to_numpy(resampled)


def augment_noise(arr: np.ndarray) -> np.ndarray:
    """Additive Gaussian noise — near-zero cost, runs live in collator."""
    snr_db    = random.uniform(*NOISE_SNR_RANGE)
    x         = torch.from_numpy(arr.astype(np.float32))
    sig_pow   = x.pow(2).mean().clamp(min=1e-9)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return (x + torch.randn_like(x) * noise_pow.sqrt()).numpy()


# Fast augmentations applied live in the collator
LIVE_AUG_FUNCTIONS = [augment_speed, augment_noise]

# ──────────────────────────────────────────────
# PREPROCESSING  (with pitch pre-baking)
#
# Cache layout:
#   ./processed_data/mHuBERT-147_5sec/original/  ← clean audio
#   ./processed_data/mHuBERT-147_5sec/pitch/     ← pitch-shifted copy
#   valid same structure (original only)
#
# Training set = original + pitch copy (2× data)
# ──────────────────────────────────────────────
keep_cols = ['speaker_id', 'language']

def preprocess_function(examples, aug_fn=None):
    arrays = [x["array"] for x in examples["audio_filepath"]]
    if aug_fn is not None:
        augmented = []
        for a in arrays:
            try:
                augmented.append(aug_fn(a))
            except Exception:
                print('Got exception in preprocess_function')
                augmented.append(a)   # fallback to original on error
        arrays = augmented

    inputs = feature_extractor(
        arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )
    inputs["label"]  = [str_to_int[x] for x in examples["language"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    inputs["length"] = [len(f) for f in inputs[input_features_key]]
    return inputs


cache_base = f"./processed_data/{model_id.split('/')[-1]}_{max_duration}sec"


def load_or_create(ds, split_name: str, aug_name: str, aug_fn=None):
    """Load from disk if cached, otherwise process in a loop and save."""
    path = f"{cache_base}/{split_name}/{aug_name}"
    bs = 64 if (aug_fn == augment_pitch) else 32  # larger batch amortises GPU launch

    if os.path.exists(path):
        print(f"  Loading cached {split_name}/{aug_name} from {path}")
        return load_from_disk(path)

    print(f"  Preprocessing {split_name}/{aug_name} "
          f"({'GPU' if aug_fn == augment_pitch else 'CPU'})…")
    processed_data = defaultdict(list)
    columns_to_remove = set(c for c in ds.column_names if c not in keep_cols)

    for i in range(0, len(ds), bs):
        batch = ds[i: i + bs]
        new_data = preprocess_function(batch, aug_fn=aug_fn)
        merged_batch = {**batch, **new_data}
        for col_name, col_values in merged_batch.items():
            if col_name not in columns_to_remove:
                processed_data[col_name].extend(col_values)

    result = Dataset.from_dict(processed_data)

    result.save_to_disk(path)
    return result


# Build train: original + pitch-shifted copy
orig_train  = load_or_create(train_ds, 'train', 'original', aug_fn=None)
pitch_train = load_or_create(train_ds, 'train', 'pitch',    aug_fn=augment_pitch)
train_ds_encoded = concatenate_datasets([orig_train, pitch_train]).shuffle(seed=42)
print(f"Train size after pitch augmentation: {len(train_ds_encoded)} "
      f"(was {len(orig_train)})")

# Validation: original only — never augmented for fair evaluation
valid_ds_encoded = load_or_create(valid_ds, 'valid', 'original', aug_fn=None)