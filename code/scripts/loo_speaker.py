import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
import evaluate

from datasets import load_dataset, Audio
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from huggingface_hub import login
import subprocess

print("===== GPU DEBUG INFO =====")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
subprocess.run(["hostname"])
subprocess.run(["nvidia-smi"])
print("==========================")

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current time: {current_time_str}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print("Working directory:", os.getcwd())
print("Files in working dir:", os.listdir(""))

# ──────────────────────────────────────────────
# LOGIN
# ──────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=HF_TOKEN)

WANDB_KEY = os.getenv("WANDB_KEY")
if not WANDB_KEY:
    raise ValueError("WANDB_KEY environment variable not set")
wandb.login(key=WANDB_KEY)

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
model_id                    = "utter-project/mHuBERT-147"
batch_size                  = 32
gradient_accumulation_steps = 4
num_train_epochs            = 35
label_smoothing             = 0.1
weight_decay                = 0.05
sampling_rate               = 16000
warmup_ratio                = 0.15
backbone_lr                 = 1e-6
head_lr                     = 1e-3
early_stopping_patience     = 8
max_grad_norm               = 0.5
max_duration                = 5
NUM_TRANSFORMER_LAYERS      = 4
AVAILABLE_CPU               = 4

unfreeze_start_step         = 300
unfreeze_every_steps        = 50

do_apply_dropout    = True
hidden_dropout      = 0.1
attention_dropout   = 0.05
activation_dropout  = 0.05
feat_proj_dropout   = 0.05

input_features_key = "input_values"

# ──────────────────────────────────────────────
# FEATURE EXTRACTOR
# ──────────────────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

# ──────────────────────────────────────────────
# DATASET + SPEAKER SPLIT
# ──────────────────────────────────────────────
dataset = load_dataset("badrex/nnti-dataset-full")
train_ds = dataset['train']

# find one speaker per language to exclude
excluded_speakers = {}
for lang in train_ds.unique('language'):
    lang_samples = train_ds.filter(lambda x, l=lang: x["language"] == l)
    excluded_speakers[lang] = lang_samples[0]["speaker_id"]

print("Excluded speakers:", excluded_speakers)

# create valid from excluded speakers
def is_excluded(example):
    return example["speaker_id"] == excluded_speakers[example["language"]]

valid_ds = dataset['train'].filter(is_excluded)

# filter excluded speakers out of train
def should_keep(example):
    return example["speaker_id"] != excluded_speakers[example["language"]]

train_ds = train_ds.filter(should_keep)

# shuffle and cast
train_ds = train_ds.shuffle(seed=42)
valid_ds = valid_ds.shuffle(seed=42)

print(f"Train size: {len(train_ds)}")
print(f"Valid size: {len(valid_ds)}")

train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

# ──────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────
LABELS        = sorted(train_ds.unique('language'))
sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages ({len(LABELS)}): {sorted_labels}")
str_to_int    = {s: i for i, s in enumerate(LABELS)}
int_to_str    = {i: s for s, i in str_to_int.items()}
num_labels    = len(LABELS)

# ──────────────────────────────────────────────
# PREPROCESSING (no augmentation)
# ──────────────────────────────────────────────
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio_filepath"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    inputs["length"] = [len(f) for f in inputs[input_features_key]]
    return inputs

keep_cols = ['speaker_id', 'language']

train_ds_encoded = train_ds.map(
    preprocess_function,
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=AVAILABLE_CPU,
)

valid_ds_encoded = valid_ds.map(
    preprocess_function,
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=AVAILABLE_CPU,
)

# ──────────────────────────────────────────────
# COLLATOR (no augmentation)
# ──────────────────────────────────────────────
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }
        batch = self.feature_extractor.pad(batch, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], dtype=torch.long)
        return batch

data_collator = AudioDataCollator(feature_extractor)

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
config = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id   = str_to_int
config.id2label   = int_to_str
if do_apply_dropout:
    config.hidden_dropout     = hidden_dropout
    config.attention_dropout  = attention_dropout
    config.activation_dropout = activation_dropout
    config.feat_proj_dropout  = feat_proj_dropout

slid_model = AutoModelForAudioClassification.from_pretrained(model_id, config=config)

slid_model.hubert.gradient_checkpointing_enable()
print("✓ Gradient checkpointing enabled")

for param in slid_model.hubert.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in slid_model.parameters() if p.requires_grad)
print(f"Trainable params at start: {trainable:,} (head only)")

# ──────────────────────────────────────────────
# WANDB INIT
# ──────────────────────────────────────────────
wandb.init(
    project="Indic-SLID",
    name=f"SLID_speaker_split_{model_id.split('/')[-1]}_{current_time_str}",
    config={
        "model_id": model_id,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "max_duration": max_duration,
        "unfreeze_start_step": unfreeze_start_step,
        "unfreeze_every_steps": unfreeze_every_steps,
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "hidden_dropout": hidden_dropout,
        "early_stopping_patience": early_stopping_patience,
        "speaker_independent_split": True,
    }
)

# ──────────────────────────────────────────────
# OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────
steps_per_epoch    = len(train_ds_encoded) // (batch_size * gradient_accumulation_steps)
num_training_steps = steps_per_epoch * num_train_epochs
warmup_steps_n     = int(warmup_ratio * num_training_steps)
last_unfreeze      = unfreeze_start_step + NUM_TRANSFORMER_LAYERS * unfreeze_every_steps

print(f"\n── Training schedule ────────────────────────")
print(f"  steps/epoch:          {steps_per_epoch}")
print(f"  total steps:          {num_training_steps}")
print(f"  warmup ends:          step {warmup_steps_n}")
print(f"  unfreeze starts:      step {unfreeze_start_step}")
print(f"  all layers active:    step {last_unfreeze}")
assert last_unfreeze < num_training_steps, "Increase num_train_epochs!"

def get_optimizer_and_scheduler(model, num_training_steps):
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and "projector" not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if "classifier" in n or "projector" in n]
    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ], weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_n,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

optimizer, scheduler = get_optimizer_and_scheduler(slid_model, num_training_steps)

# ──────────────────────────────────────────────
# TRAINING ARGS
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="/home/neuronet_team108/NNTIProject/indic-SLID/checkpoints",
    group_by_length=True,
    dataloader_num_workers=AVAILABLE_CPU,
    dataloader_pin_memory=True,
    report_to="wandb",
    logging_steps=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps_n,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,
    fp16=True,
    max_grad_norm=max_grad_norm,
    push_to_hub=False,
    label_smoothing_factor=label_smoothing,
    remove_unused_columns=False,
)

# ──────────────────────────────────────────────
# METRICS & CALLBACKS
# ──────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)


class GradualUnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_start_step, unfreeze_every_steps):
        self.unfreeze_start_step  = unfreeze_start_step
        self.unfreeze_every_steps = unfreeze_every_steps
        self.layers_unfrozen      = 0
        self._final_unfreeze_done = False

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        step   = state.global_step
        if step < self.unfreeze_start_step:
            return
        layers = model.hubert.encoder.layers
        n      = len(layers)
        target = (step - self.unfreeze_start_step) // self.unfreeze_every_steps + 1

        if self.layers_unfrozen < target <= n:
            layer_idx = n - target
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
            self.layers_unfrozen = target
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ Unfrozen layer {layer_idx} at step {step} "
                  f"({self.layers_unfrozen}/{n}) | trainable: {trainable:,}")

        if self.layers_unfrozen >= n and not self._final_unfreeze_done:
            for name, param in model.hubert.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True
            self._final_unfreeze_done = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ Phase 3: all transformer params active at step {step} "
                  f"| trainable: {trainable:,}")

# ──────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────
trainer = Trainer(
    slid_model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    callbacks=[
        GradualUnfreezeCallback(
            unfreeze_start_step=unfreeze_start_step,
            unfreeze_every_steps=unfreeze_every_steps,
        ),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ],
)

# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
torch.cuda.empty_cache()
free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
print(f"\nVRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

print("Train loop starting...")
trainer.train()

# ──────────────────────────────────────────────
# EVALUATE & LOG
# ──────────────────────────────────────────────
print("Final evaluation...")
metrics = trainer.evaluate()
wandb.run.summary["eval_accuracy"] = metrics["eval_accuracy"]

predictions = trainer.predict(valid_ds_encoded)
preds  = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None, y_true=labels.tolist(), preds=preds.tolist(),
    class_names=sorted_labels,
)})

# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────
save_dir = f"/home/neuronet_team108/NNTIProject/indic-SLID/best_Hubert_{current_time_str}"
trainer.save_model(save_dir)
print(f"Model saved to {save_dir}")

wandb.finish()