import os
from datetime import datetime
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import Counter
import random
import pandas as pd
import numpy as np
import torch
import torchaudio.transforms as T
import wandb
from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor, 
    Wav2Vec2ForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    set_seed
)
from datasets import (
    load_dataset, 
    Audio
    # load_from_disk, 
    # DatasetDict, 
    # concatenate_datasets, 
)
from huggingface_hub import login
# import Hugging Face libraries
import evaluate

print("===== GPU DEBUG INFO =====")

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

print("\n--- hostname ---")
subprocess.run(["hostname"])

print("\n--- nvidia-smi ---")
subprocess.run(["nvidia-smi"])

print("\n--- ECC errors ---")
subprocess.run(["nvidia-smi", "-q", "-d", "ECC"])

print("==========================")

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Current time: {current_time_str}")
print("Torch:", torch.__version__)
print("wandb:", wandb.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("==========================")

# %%
# check if there GPU
print("Check if GPU available:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")

# %%
# login to Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=HF_TOKEN)

# %%
# login to WANDB
WANDB_KEY = os.getenv("WANDB_KEY")
if not WANDB_KEY:
    raise ValueError("WANDB_KEY environment variable not set")
wandb.login(key=WANDB_KEY)

# %%

model_path = "./inprogress_tjvqnpsn"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/mms-300m", 
    do_normalize=True,
    return_attention_mask=True,
)

# %%
dataset = load_dataset("badrex/nnti-dataset-full")

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
max_duration = 7 # in seconds

# %%
# get the set of languages
LABELS = train_ds.unique('language')

sorted_labels = sorted(l.upper() for l in LABELS) 
print(f"Languages: {sorted_labels}")

str_to_int = {
    s: i for i, s in enumerate(LABELS)
}


# %%
def preprocess_function(examples):

    audio_arrays = [] #(OLD)[x["array"] for x in examples["audio_filepath"]]
    
    # Add stochastic augmentation to each audio sample in the batch
    for x in examples["audio_filepath"]:
        # Load waveform (ensure it's 16kHz)
        waveform = torch.tensor(x["array"]).unsqueeze(0).float()
            
        # Stochastic Augmentation (50% chance)
        if random.random() > 0.5:
            # Randomly choose between Pitch Shift or Speed Change
            choice = random.choice(["pitch", "speed"])
            
            if choice == "pitch":
                # Random shift between -2 and 2 semitones
                n_steps = random.randint(-2, 2)
                aug = T.PitchShift(16000, n_steps)
                waveform = aug(waveform)
            
            elif choice == "speed":
                # Speed up or slow down by 10%
                factor = random.choice([0.9, 1.1])
                aug = T.SpeedPerturbation(16000, [factor])
                waveform, _ = aug(waveform)

        audio_arrays.append(waveform.squeeze(0).detach().numpy())
    

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    
    # convert input_features_key contains numerical arrays
    inputs["input_values"] = [
        np.array(x) for x in inputs["input_values"]
    ]

    inputs["length"] = [len(f) for f in inputs["input_values"]]
    
    return inputs

def preprocess_function_val(examples):

    audio_arrays = [x["array"] for x in examples["audio_filepath"]]

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    
    # convert input_features_key contains numerical arrays
    inputs["input_values"] = [
        np.array(x) for x in inputs["input_values"]
    ]

    inputs["length"] = [len(f) for f in inputs["input_values"]]
    return inputs

# %%
keep_cols = ['speaker_id', 'language']

# %% [markdown]
# ## encode the train and valid splits 

# %%
train_ds_encoded = train_ds.map(
    preprocess_function, 
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=5,
)

# %%
valid_ds_encoded = valid_ds.map(
    preprocess_function_val, 
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=5,
)

# %%
int_to_str = {
    i: s for s, i in str_to_int.items()
}

num_labels = len(int_to_str)

# %%
config = AutoConfig.from_pretrained(model_path)
config.num_labels=num_labels
config.label2id=str_to_int
config.id2label=int_to_str

model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, config=config)


# %%
# create collator for padding
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare the batch dict in the format expected by the feature extractor
        batch = {
            "input_values": [f["input_values"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }
        
        # use the feature extractor's native padding
        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        
        # add labels
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], 
            dtype=torch.long
        )
        
        return batch


# %%
data_collator = AudioDataCollator(feature_extractor)

# %%
batch_size = 8
gradient_accumulation_steps = 8
num_train_epochs = 10
lr = 0.00002646029450927161

# %%
wandb.init(project="Indic-SLID", name=f"SLID_{'Best_Model'}_{lr}_{current_time_str}")
    
training_args = TrainingArguments(
    output_dir="./indic-SLID/checkpoints",
    group_by_length=False,
    #run_name='SLID_1', 
    report_to="wandb",  # enable logging to W&B
    logging_steps=10,  # how often to log to W&B
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,    
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps", 
    save_steps=100,
    learning_rate=lr,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=0.1755957898788224,
    warmup_ratio=0.13520028959798058,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,  # True if your metric should be maximized (like accuracy)
    save_total_limit=2,  # Keep only the best model
    fp16=True,
    push_to_hub=False,
)
# %%
# load evaluation metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(
        predictions=predictions, 
        references=eval_pred.label_ids
    )

# %%
trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

# %%
print("Train loop starting...")
trainer.train()

# final eval and save for each trial
print("Final evaluation starting...")
trainer.evaluate()

save_dir = f"./indic-SLID/inprogress_{current_time_str}"  # unique per trial
trainer.save_model(save_dir)

wandb.finish()




