"""
kNN-VC Preprocessing Script for M1 Mac
================================================
Fix: kNN-VC runs in the MAIN process — no forking, no MPS crash.
     We process samples manually instead of using ds.map().

Usage:
    pip install pyarrow torch numpy
    export HF_TOKEN=your_token_here
    python preprocess_knnvc.py

Output:
    ./knnvc_cache/mHuBERT-147_5sec_knnvc.zip
"""

import os
import sys
import random
import zipfile
import time
import gc
import shutil
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────
# SETTINGS  (must match train_knnvc.py exactly)
# ──────────────────────────────────────────────
MODEL_ID              = "utter-project/mHuBERT-147"
MAX_DURATION          = 5
SAMPLING_RATE         = 16000
KNN_TOPK              = 4
KNN_COPIES_PER_SAMPLE = 1

OUTPUT_DIR = Path("./knnvc_cache")
CACHE_BASE = OUTPUT_DIR / f"{MODEL_ID.split('/')[-1]}_{MAX_DURATION}sec_knnvc"
ZIP_OUTPUT = OUTPUT_DIR / f"{MODEL_ID.split('/')[-1]}_{MAX_DURATION}sec_knnvc.zip"

INPUT_FEATURES_KEY = "input_values"
KEEP_COLS          = ['speaker_id', 'language']

# M1: use MPS in main process only — never pass to subprocesses
try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✓ MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("✓ CUDA")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  CPU only")
except Exception as e:
    DEVICE = torch.device("cpu")
    print(f"⚠️  MPS check failed, defaulting to CPU: {e}")

import multiprocessing
NUM_PROC = multiprocessing.cpu_count() - 2
print(f"CPU cores: {NUM_PROC} | Device: {DEVICE}")

from datasets import load_dataset, Audio, load_from_disk, Dataset, concatenate_datasets
from transformers import AutoFeatureExtractor
from huggingface_hub import login


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
def load_data():
    print("\n── Loading dataset ──────────────────────────")
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    if os.path.exists('./nnti-dataset'):
        print("Loading dataset from disk cache…")
        dataset = load_from_disk('./nnti-dataset')
    else:
        print("Loading dataset from Hugging Face Hub…")
        dataset = load_dataset('badrex/nnti-dataset-full')
        dataset.save_to_disk('./nnti-dataset')
    train_ds = dataset['train'].shuffle(seed=42).cast_column(
        "audio_filepath", Audio(sampling_rate=SAMPLING_RATE))
    valid_ds = dataset['validation'].shuffle(seed=42).cast_column(
        "audio_filepath", Audio(sampling_rate=SAMPLING_RATE))

    LABELS     = sorted(train_ds.unique('language'))
    str_to_int = {s: i for i, s in enumerate(LABELS)}
    print(f"Languages ({len(LABELS)}): {sorted(l.upper() for l in LABELS)}")
    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")
    return train_ds, valid_ds, str_to_int


# ──────────────────────────────────────────────
# kNN-VC SETUP
# ──────────────────────────────────────────────
def setup_knn_vc_repo():
    repo = Path("./knn-vc")
    if not repo.exists():
        print("Cloning kNN-VC...")
        os.system("git clone --quiet https://github.com/bshall/knn-vc.git ./knn-vc")
    os.system("pip install -q -r ./knn-vc/requirements.txt")
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    print("✓ kNN-VC repo ready")


def load_knn_vc():
    setup_knn_vc_repo()
    from hubconf import knn_vc as load_fn
    kvc = load_fn(prematched=True, device=DEVICE)
    print(f"✓ kNN-VC loaded on {DEVICE}")
    return kvc


# ──────────────────────────────────────────────
# AUGMENTER
# ──────────────────────────────────────────────
class KNNVCAugmenter:
    MIN_SAMPLES = 16000  # 1 секунда — надёжный минимум для WavLM

    def __init__(self, raw_train_ds, kvc, topk: int = KNN_TOPK):
        self.kvc         = kvc
        self.topk        = topk
        self._feat_cache = {}
        self._spk_rows   = {}
        self._skip_count = 0

        for i, spk in enumerate(raw_train_ds['speaker_id']):
            self._spk_rows.setdefault(spk, []).append(i)
        self.all_speakers = list(self._spk_rows.keys())
        self._raw_ds      = raw_train_ds
        print(f"  Indexed {len(self.all_speakers)} speakers")

    def _get_matching_set(self, speaker_id: str) -> torch.Tensor:
        if speaker_id in self._feat_cache:
            return self._feat_cache[speaker_id]
        rows = random.sample(
            self._spk_rows[speaker_id],
            min(5, len(self._spk_rows[speaker_id]))
        )
        wavs = [
            torch.tensor(
                self._raw_ds[i]['audio_filepath']['array'],
                dtype=torch.float32
            )
            for i in rows
        ]
        with torch.no_grad():
            # get_matching_set принимает list of waveforms → возвращает (N_frames, 1024)
            feats = self.kvc.get_matching_set(wavs)
        self._feat_cache[speaker_id] = feats
        return feats

    def convert(self, source_array: np.ndarray, source_speaker_id: str) -> np.ndarray:
        if len(source_array) < self.MIN_SAMPLES:
            self._skip_count += 1
            return source_array

        try:
            other = [s for s in self.all_speakers if s != source_speaker_id]
            tgt_spk = random.choice(other)
            mset = self._get_matching_set(tgt_spk)
            src_wav = torch.tensor(source_array, dtype=torch.float32)

            with torch.no_grad():
                query_features = self.kvc.get_features(src_wav.unsqueeze(0))

            # Guard against any empty output regardless of input length
            if query_features.numel() == 0 or query_features.shape[0] == 0:
                self._skip_count += 1
                return source_array

            with torch.no_grad():
                out = self.kvc.match(
                    query_seq=query_features,
                    matching_set=mset,
                    topk=self.topk,
                )

            result = out.cpu().numpy().astype(np.float32)
            if len(result) < len(source_array):
                result = np.pad(result, (0, len(source_array) - len(result)))
            return result[:len(source_array)]

        except Exception as e:
            self._skip_count += 1
            return source_array


# ──────────────────────────────────────────────
# PLAIN PREPROCESSING  (CPU only → safe to fork with ds.map)
# ──────────────────────────────────────────────
def make_preprocess_plain(feature_extractor, str_to_int):
    def fn(examples):
        arrays = [x["array"] for x in examples["audio_filepath"]]
        inputs = feature_extractor(
            arrays,
            sampling_rate=SAMPLING_RATE,
            truncation=True,
            max_length=SAMPLING_RATE * MAX_DURATION,
            return_attention_mask=True,
        )
        inputs["label"]            = [str_to_int[x] for x in examples["language"]]
        inputs[INPUT_FEATURES_KEY] = [np.array(x) for x in inputs[INPUT_FEATURES_KEY]]
        inputs["length"]           = [len(f) for f in inputs[INPUT_FEATURES_KEY]]
        return inputs
    return fn


def load_or_create_plain(ds, split_name, feature_extractor, str_to_int):
    path = CACHE_BASE / split_name / "original"
    if path.exists():
        print(f"  ✓ Cache hit: {split_name}/original")
        return load_from_disk(str(path))
    print(f"  Preprocessing {split_name}/original (CPU × {NUM_PROC})…")
    t0     = time.time()
    fn     = make_preprocess_plain(feature_extractor, str_to_int)
    result = ds.map(
        fn, batched=True, batch_size=32, num_proc=NUM_PROC,
        remove_columns=[c for c in ds.column_names if c not in KEEP_COLS],
    )
    result.save_to_disk(str(path))
    print(f"  ✓ Done in {timedelta(seconds=int(time.time()-t0))}")
    return result


# ──────────────────────────────────────────────
# kNN-VC PREPROCESSING — MAIN PROCESS ONLY, NO FORKING
#
# Root cause of the crash:
#   datasets.map() forks a subprocess even with num_proc=1.
#   MPS (and CUDA) contexts don't survive fork → immediate crash.
#
# Fix:
#   Iterate samples manually in the main process.
#   Accumulate converted waveforms → batch feature extraction (CPU) →
#   write parquet chunks → reassemble into HuggingFace Dataset.
# ──────────────────────────────────────────────
def create_knnvc_dataset(
    ds,
    split_name: str,
    copy_idx: int,
    augmenter: KNNVCAugmenter,
    feature_extractor,
    str_to_int: dict,
) -> Dataset:
    path = CACHE_BASE / split_name / f"knnvc_{copy_idx}"
    if path.exists():
        print(f"  ✓ Cache hit: {split_name}/knnvc_{copy_idx}")
        return load_from_disk(str(path))

    n       = len(ds)
    max_len = SAMPLING_RATE * MAX_DURATION
    BATCH   = 16   # samples per parquet chunk (adjust if RAM is tight)

    print(f"\n  🎙  kNN-VC: {split_name}/knnvc_{copy_idx} ({n} samples)")
    print(f"  Running in main process on {DEVICE} — no subprocess forking")

    tmp_path  = path.parent / f"knnvc_{copy_idx}_tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)
    chunk_files  = []

    # Per-batch accumulators
    buf_arrays   = []
    buf_labels   = []
    buf_langs    = []
    buf_speakers = []

    t0          = time.time()
    last_report = t0
    chunk_idx   = 0

    def flush(cidx):
        """Feature-extract current buffer and write one parquet chunk."""
        if not buf_arrays:
            return
        inputs = feature_extractor(
            buf_arrays,
            sampling_rate=SAMPLING_RATE,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            padding=False,
        )
        table = pa.table({
            INPUT_FEATURES_KEY: pa.array(
                [np.array(x, dtype=np.float32) for x in inputs[INPUT_FEATURES_KEY]],
                type=pa.list_(pa.float32())
            ),
            "attention_mask": pa.array(
                [np.array(x, dtype=np.int32) for x in inputs["attention_mask"]],
                type=pa.list_(pa.int32())
            ),
            "label":      pa.array(buf_labels,   type=pa.int64()),
            "length":     pa.array(
                [min(len(a), max_len) for a in buf_arrays], type=pa.int64()
            ),
            "language":   pa.array(buf_langs,    type=pa.string()),
            "speaker_id": pa.array(buf_speakers, type=pa.string()),
        })
        out = tmp_path / f"chunk_{cidx:05d}.parquet"
        pq.write_table(table, str(out))
        chunk_files.append(out)
        buf_arrays.clear(); buf_labels.clear()
        buf_langs.clear();  buf_speakers.clear()

    for i in range(n):
        sample    = ds[i]
        arr       = sample['audio_filepath']['array'].astype(np.float32)
        spk_id    = sample['speaker_id']
        lang      = sample['language']
        converted = augmenter.convert(arr, spk_id)

        buf_arrays.append(converted)
        buf_labels.append(str_to_int[lang])
        buf_langs.append(lang)
        buf_speakers.append(spk_id)

        if len(buf_arrays) >= BATCH:
            flush(chunk_idx)
            chunk_idx += 1

        # Progress every 60s
        now = time.time()
        if now - last_report > 60:
            pct     = (i + 1) / n * 100
            pct_skipped = augmenter._skip_count / (i + 1) * 100
            elapsed = now - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n} ({pct:.1f}%) | "
                  f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                  f"ETA: {timedelta(seconds=int(eta))} |"
                  f"Skipped: {augmenter._skip_count} ({pct_skipped:.1f}%)")
            last_report = now

    flush(chunk_idx)   # remaining samples

    elapsed = timedelta(seconds=int(time.time() - t0))
    print(f"  ✓ Conversion done in {elapsed}. Assembling dataset…")

    # Merge parquet chunks → HuggingFace Dataset
    tables  = [pq.read_table(str(f)) for f in sorted(chunk_files)]
    merged  = pa.concat_tables(tables)
    dataset = Dataset(merged)
    dataset.save_to_disk(str(path))
    shutil.rmtree(str(tmp_path))

    print(f"  ✓ Saved {len(dataset)} samples → {path}")
    return dataset


# ──────────────────────────────────────────────
# ZIP
# ──────────────────────────────────────────────
def create_zip():
    print(f"\n── Creating zip ─────────────────────────────")
    t0 = time.time()
    count = 0
    with zipfile.ZipFile(ZIP_OUTPUT, 'w',
                         compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for fpath in sorted(CACHE_BASE.rglob("*")):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(OUTPUT_DIR))
                count += 1
                if count % 500 == 0:
                    print(f"  {count} files… ({ZIP_OUTPUT.stat().st_size/1e6:.0f}MB)")
    size_gb = ZIP_OUTPUT.stat().st_size / 1e9
    print(f"  ✓ {count} files → {size_gb:.2f}GB in {timedelta(seconds=int(time.time()-t0))}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  kNN-VC Preprocessing — M1 Mac (fixed)")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_BASE.mkdir(parents=True, exist_ok=True)

    train_ds, valid_ds, str_to_int = load_data()
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, do_normalize=True, return_attention_mask=True
    )

    # Step 1: plain (multi-CPU, no GPU, safe to fork)
    print("\n── Step 1/3: Plain preprocessing ───────────")
    orig_train       = load_or_create_plain(train_ds, 'train', feature_extractor, str_to_int)
    valid_ds_encoded = load_or_create_plain(valid_ds, 'valid', feature_extractor, str_to_int)

    # Step 2: kNN-VC (main process loop, no fork)
    print("\n── Step 2/3: kNN-VC augmentation ───────────")
    missing = [
        i for i in range(KNN_COPIES_PER_SAMPLE)
        if not (CACHE_BASE / 'train' / f'knnvc_{i}').exists()
    ]
    knnvc_parts = [orig_train]

    if missing:
        print("  Loading kNN-VC model…")
        kvc       = load_knn_vc()
        augmenter = KNNVCAugmenter(train_ds, kvc, topk=KNN_TOPK)
        for i in range(KNN_COPIES_PER_SAMPLE):
            knnvc_parts.append(
                create_knnvc_dataset(
                    train_ds, 'train', i, augmenter, feature_extractor, str_to_int
                )
            )
        del augmenter, kvc
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        print("  ✓ kNN-VC freed")
    else:
        print("  All copies cached — skipping model load")
        for i in range(KNN_COPIES_PER_SAMPLE):
            knnvc_parts.append(
                load_from_disk(str(CACHE_BASE / 'train' / f'knnvc_{i}'))
            )

    combined = concatenate_datasets(knnvc_parts)
    print(f"\n  Train: {len(combined)} ({len(knnvc_parts)}× original {len(orig_train)})")
    print(f"  Valid: {len(valid_ds_encoded)}")

    # Step 3: zip
    print("\n── Step 3/3: Zipping ────────────────────────")
    create_zip()

    print("\n" + "=" * 55)
    print(f"  ✓ Done!  {ZIP_OUTPUT.resolve()}")
    print()
    print("  Unzip in Colab:")
    print("    from google.colab import drive")
    print("    drive.mount('/content/drive')")
    print("    import zipfile")
    print(f"    with zipfile.ZipFile('/content/drive/MyDrive/{ZIP_OUTPUT.name}') as z:")
    print("        z.extractall('./processed_data')")
    print("  Set CACHE_ROOT = './processed_data' in train_knnvc.py")
    print("=" * 55)


if __name__ == "__main__":
    main()