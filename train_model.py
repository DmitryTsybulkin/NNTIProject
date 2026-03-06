# %%
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import wandb
import evaluate

from datasets import load_from_disk, concatenate_datasets
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
from huggingface_hub import login, snapshot_download
import random

if not os.getenv('RUNNING_ON_CLUSTER'):
    from dotenv import load_dotenv
    load_dotenv()

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current time: {current_time_str}")

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")

import multiprocessing

# TODO: change for the certain environment
AVAILABLE_CPU = multiprocessing.cpu_count()
WANDB_TEAM    = os.getenv("WANDB_TEAM")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
HF_TOKEN      = os.getenv('HF_TOKEN')
WANDB_TOKEN   = os.getenv('WANDB_TOKEN')


login(token=HF_TOKEN)
wandb.login(key=WANDB_TOKEN)

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
# batch_size=64 + grad_checkpointing fits safely (Should work for T4 and P100 gpu).
# effective_batch = 64 * 2 = 128 (same as before -> same convergence dynamics)
#
# mHuBERT-147 has 12 transformer layers.
#
# Unfreeze timeline (steps):
#   0    -> 506:   warmup, head only
#   556  -> 1756:  1 transformer layer per 100 steps (top -> bottom)
#   1756 -> 3375:  full model (except CNN) — ~12 epoch buffer

model_id                    = "utter-project/mHuBERT-147"
batch_size                  = 64
gradient_accumulation_steps = 2      # effective batch = 128
num_train_epochs            = 25
label_smoothing             = 0.1
weight_decay                = 0.05
sampling_rate               = 16000
warmup_ratio                = 0.15
backbone_lr                 = 0.000001
head_lr                     = 0.001
early_stopping_patience     = 8
max_grad_norm               = 0.5
max_duration                = 5
NUM_TRANSFORMER_LAYERS      = 12

unfreeze_start_step         = 600
unfreeze_every_steps        = 100

do_apply_dropout    = True
hidden_dropout      = 0.1
attention_dropout   = 0.01
activation_dropout  = 0.01
feat_proj_dropout   = 0.0

AUG_PROB         = 0.5
NOISE_SNR_RANGE  = (15.0, 40.0)
input_features_key = "input_values"

KNNVC_CACHE_REPO = "radioduran/mHuBERT-knnvc-dataset"
KNNVC_LOCAL_DIR  = "./knnvc_cache_download"

wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_TEAM,
    name=f"SLID_knnvc_{model_id.split('/')[-1]}_{current_time_str}",
    config={
        "model_id":                model_id,
        "batch_size":              batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size":    batch_size * gradient_accumulation_steps,
        "num_train_epochs":        num_train_epochs,
        "backbone_lr":             backbone_lr,
        "head_lr":                 head_lr,
        "max_duration":            max_duration,
        "unfreeze_start_step":     unfreeze_start_step,
        "unfreeze_every_steps":    unfreeze_every_steps,
        "num_transformer_layers":  NUM_TRANSFORMER_LAYERS,
        "label_smoothing":         label_smoothing,
        "weight_decay":            weight_decay,
        "warmup_ratio":            warmup_ratio,
        "max_grad_norm":           max_grad_norm,
        "hidden_dropout":          hidden_dropout,
        "aug_strategy":            "knnvc+speed+noise",
        "aug_prob":                AUG_PROB,
        "early_stopping_patience": early_stopping_patience,
        "feature_extractor_frozen": True,
    }
)

# ──────────────────────────────────────────────
# FEATURE EXTRACTOR
# ──────────────────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

# ──────────────────────────────────────────────
# LOAD kNN-VC CACHE FROM HUGGING FACE
# ──────────────────────────────────────────────
cache_base = os.path.join(
    KNNVC_LOCAL_DIR, "knnvc_cache",
    f"{model_id.split('/')[-1]}_{max_duration}sec_knnvc",
)

if not os.path.exists(KNNVC_LOCAL_DIR):
    print(f"Downloading kNN-VC cache from {KNNVC_CACHE_REPO} ...")
    snapshot_download(
        repo_id=KNNVC_CACHE_REPO,
        repo_type="dataset",
        local_dir=KNNVC_LOCAL_DIR,
        token=HF_TOKEN,
    )
    print("Download complete")
else:
    print(f"Using cached download at {KNNVC_LOCAL_DIR}")

print("Loading Arrow datasets...")
orig_train       = load_from_disk(os.path.join(cache_base, "train", "original"))
knnvc_train      = load_from_disk(os.path.join(cache_base, "train", "knnvc_0"))
valid_ds_encoded = load_from_disk(os.path.join(cache_base, "valid", "original"))

train_ds_encoded = concatenate_datasets([orig_train, knnvc_train]).shuffle(seed=42)
print(f"Train: {len(train_ds_encoded)} ({len(orig_train)} orig + {len(knnvc_train)} knnvc)")
print(f"Valid: {len(valid_ds_encoded)}")

# ──────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────
LABELS        = sorted(set(orig_train['language']))
sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages ({len(LABELS)}): {sorted_labels}")
str_to_int    = {s: i for i, s in enumerate(LABELS)}
int_to_str    = {i: s for s, i in str_to_int.items()}
num_labels    = len(LABELS)

# Verify cached integer labels match our str_to_int mapping
sample_lang  = orig_train[0]['language']
sample_label = orig_train[0]['label']
REMAP_LABELS = (sample_label != str_to_int[sample_lang])
if REMAP_LABELS:
    print(f"⚠️  Label remapping needed (cached {sample_label} != expected {str_to_int[sample_lang]})")
else:
    print(f"✅ Labels consistent")

# ──────────────────────────────────────────────
# LIVE AUGMENTATION
# ──────────────────────────────────────────────
def _to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

def _to_numpy(t):
    return t.squeeze(0).numpy()

SPEED_RATES = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
RESAMPLERS  = {
    rate: T.Resample(orig_freq=int(sampling_rate * rate), new_freq=sampling_rate)
    for rate in SPEED_RATES
}

def augment_speed(arr):
    rate      = random.choice(SPEED_RATES)
    resampled = RESAMPLERS[rate](_to_tensor(arr))
    orig_len  = arr.shape[0]
    if resampled.shape[1] < orig_len:
        resampled = F.pad(resampled, (0, orig_len - resampled.shape[1]))
    else:
        resampled = resampled[:, :orig_len]
    return _to_numpy(resampled)

def augment_noise(arr):
    snr_db    = random.uniform(*NOISE_SNR_RANGE)
    x         = torch.from_numpy(arr.astype(np.float32))
    sig_pow   = x.pow(2).mean().clamp(min=1e-9)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return (x + torch.randn_like(x) * noise_pow.sqrt()).numpy()

LIVE_AUG_FUNCTIONS = [augment_speed, augment_noise]

# ──────────────────────────────────────────────
# COLLATOR
# ──────────────────────────────────────────────
class AugmentingCollator:
    def __init__(self, aug_prob=0.5, training=True):
        self.aug_prob = aug_prob
        self.training = training

    def _maybe_augment(self, arr):
        if not self.training or random.random() > self.aug_prob:
            return arr
        try:
            return random.choice(LIVE_AUG_FUNCTIONS)(arr)
        except Exception:
            return arr

    def __call__(self, features):
        augmented = [self._maybe_augment(np.array(f[input_features_key])) for f in features]
        normed = []
        for a in augmented:
            m, s = a.mean(), a.std()
            normed.append(((a - m) / (s + 1e-7)).astype(np.float32))

        batch = feature_extractor.pad(
            {input_features_key: normed,
             "attention_mask":   [f["attention_mask"] for f in features]},
            padding=True, return_tensors="pt",
        )
        if REMAP_LABELS:
            batch["labels"] = torch.tensor(
                [str_to_int[f["language"]] for f in features], dtype=torch.long)
        else:
            batch["labels"] = torch.tensor(
                [f["label"] for f in features], dtype=torch.long)
        return batch

train_collator = AugmentingCollator(aug_prob=AUG_PROB, training=True)
eval_collator  = AugmentingCollator(aug_prob=0.0,      training=False)

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

# Gradient checkpointing is essential on P100 16GB
slid_model.hubert.gradient_checkpointing_enable()
print("✅ Gradient checkpointing enabled")

# Freeze entire backbone (head is trainable by default)
for param in slid_model.hubert.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in slid_model.parameters() if p.requires_grad)
print(f"Trainable params at start: {trainable:,} (head only)")

# ──────────────────────────────────────────────
# OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────
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
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

# ──────────────────────────────────────────────
# TRAINING ARGS
# ──────────────────────────────────────────────
steps_per_epoch    = len(train_ds_encoded) // (batch_size * gradient_accumulation_steps)
num_training_steps = steps_per_epoch * num_train_epochs
warmup_steps_n     = int(warmup_ratio * num_training_steps)
last_unfreeze      = unfreeze_start_step + NUM_TRANSFORMER_LAYERS * unfreeze_every_steps

print(f"\n── Training schedule ────────────────────────")
print(f"  steps/epoch:     {steps_per_epoch}")
print(f"  total steps:     {num_training_steps}")
print(f"  warmup ends:     step {warmup_steps_n}")
print(f"  unfreeze starts: step {unfreeze_start_step}")
print(f"  all 12 layers active: step {last_unfreeze} "
      f"(epoch {last_unfreeze/steps_per_epoch:.1f})")
print(f"  buffer: {num_training_steps - last_unfreeze} steps "
      f"({(num_training_steps - last_unfreeze)/steps_per_epoch:.1f} epochs)")
assert last_unfreeze < num_training_steps, "Increase num_train_epochs!"

optimizer, scheduler = get_optimizer_and_scheduler(slid_model, num_training_steps)

training_args = TrainingArguments(
    output_dir="./indic-SLID/checkpoints",
    group_by_length=True,
    dataloader_num_workers=min(4, AVAILABLE_CPU),
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
    """
    Phase 1 (steps 0 → unfreeze_start):
        Only classification head trains. Backbone fully frozen.

    Phase 2 (unfreeze_start → unfreeze_start + 12×100):
        Unfreeze 1 transformer layer per 100 steps, top-down.
        Layer 11 first (closest to head), layer 0 last.

    Phase 3 (remaining steps):
        All transformer layers + positional embeddings + layer norms active.
        Feature Extractor CNN stays frozen permanently.
    """
    def __init__(self, unfreeze_start_step, unfreeze_every_steps):
        self.unfreeze_start_step  = unfreeze_start_step
        self.unfreeze_every_steps = unfreeze_every_steps
        self.layers_unfrozen      = 0
        self._final_unfreeze_done = False

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step < self.unfreeze_start_step:
            return

        layers  = model.hubert.encoder.layers
        n       = len(layers)
        target  = (step - self.unfreeze_start_step) // self.unfreeze_every_steps + 1

        # Unfreeze next transformer layer
        if self.layers_unfrozen < target <= n:
            layer_idx = n - target
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
            self.layers_unfrozen = target
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ Unfrozen transformer layer {layer_idx} at step {step} "
                  f"({self.layers_unfrozen}/{n}) | trainable: {trainable:,}")

        # After all transformer layers: also unfreeze positional + layer norms
        # (everything in hubert EXCEPT feature_extractor)
        if self.layers_unfrozen >= n and not self._final_unfreeze_done:
            for name, param in model.hubert.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True
            self._final_unfreeze_done = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ Phase 3: all transformer params active at step {step} "
                  f"| trainable: {trainable:,}")
            print(f"  Feature Extractor CNN remains frozen.")

# ──────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────
trainer = Trainer(
    slid_model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=train_collator,
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

original_get_eval_dataloader = trainer.get_eval_dataloader
def get_eval_dataloader_no_aug(eval_dataset=None):
    trainer.data_collator = eval_collator
    dl = original_get_eval_dataloader(eval_dataset)
    trainer.data_collator = train_collator
    return dl
trainer.get_eval_dataloader = get_eval_dataloader_no_aug

# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
torch.cuda.empty_cache()
free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
print(f"\nVRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
print(f"Model device: {slid_model.device}")

wandb.watch(slid_model, log="gradients", log_freq=100)
print("Train loop starting...")
trainer.train()

# ──────────────────────────────────────────────
# EVALUATE & LOG
# ──────────────────────────────────────────────
print("Final evaluation...")
trainer.evaluate()

predictions = trainer.predict(valid_ds_encoded)
preds  = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None, y_true=labels, preds=preds, class_names=sorted_labels,
)})

# ──────────────────────────────────────────────
# SAVE & PUSH
# ──────────────────────────────────────────────
# slid_model.push_to_hub("radioduran/indic-language-identification")
# print('Model pushed')