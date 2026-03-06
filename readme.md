# NNTI Project

This project is an attempt to adapt a pretrained speech transformer to classify and differentiate between 22 languages 
of the Indian subcontinent. We reproduced and improved the base line accuracy from 18\% to 46.667\% by experimenting 
with three different pretrained models and by using Bayesian Hyperparameter Optimization to select the best combination 
of hyperparameters. To correct speaker biases, we conducted data augmentation and k-NN voice conversion and could 
conclude that k-NN conversion was the most effective in reducing bias. This augmentation gave us 52.424\% accuracy. 
Our results showed that while a good hyperparameter selection is essential for maximizing model performance, 
its effectiveness is limited by the underlying quality of the data. It cannot compensate for structural deficiencies 
such as speaker bias. 

# Requirements:
The easiest way to run the code it to use Google Colab or any server supporting Jupyter notebooks.
The baseline pipeline with hyperparameters search were applied on Colab with Tesla T4 GPU, the second task with speaker 
bias correction was applied on L4 GPU. It's also possible to run the code on a local machine or HTCondor cluster, 
but make sure to have a compatible GPU and enough resources. For easier setup we made 2 scripts:
1. `deploy.sh` - builds the docker image and pushes it to the registry;
2. `sync_job.sh` - synchronizes the code with the cluster and submits the job.

To run these scripts it's important to have Docker installed and configured the .env file according to the .env.example.

# How to Run Standalone Code Locally

- `pip install -r requirements.txt` - to install the required dependencies;
- `python train_model.py` - to run the training script. Make sure to set the environment variables according to the .env.example before running the script.

This script contain the best model with speed+noise+kNN augmentations and the best hyperparameters found in the hyperparameter search.

# How to Reproduce Our Runs
## Task 1 - Baseline Pipeline

1. Copy `notebooks/NNTI_Hubert_Baseline.ipynb` or `notebooks/NNTI_MMS300_Baseline.ipynb` on in a configured environment
(Colab/local machine/HTCondor cluster);
2. Initialize environment variables according to the .env.example;
3. Run the notebook and wait for the results.

## Task 1 - Hyperparameter Search

1. Copy the `notebooks/NNTI_Hubert_Best_Params.ipynb` to the configured environment (Colab/local machine/HTCondor cluster).;
2. Initialize environment variables according to the .env.example. If needed - change the hyperparameter config;
3. Run the notebook and wait for the results.

## Task 2 - Speaker Bias Correction with simple augmentations

1. Copy the `notebooks/NNTI_Hubert_AugSimple.ipynb` to the configured environment (Colab/local machine/HTCondor cluster);
2. Initialize environment variables according to the .env.example;
3. Run the notebook and wait for the results of the training with "speed+pitch+noise" augmentations.

> Note: Pitch preprocessing requires more resources and time, we moved preprocessing to GPU, it's better to have at least L4 GPU.

## Task 2 - Speaker Bias Correction with k-NN Voice Conversion

1. Copy the `notebooks/NNTI_Hubert_AugKNN.ipynb` to the configured environment (Colab/local machine/HTCondor cluster);
2. Initialize environment variables according to the .env.example;
3. Run the notebook and wait for the results of the training with "speed+noise+kNN" augmentations.

> Note: for this task we prepared the k-NN dataset that can be downloaded from HuggingFace.

## Task 3 - Analyze the Model

1. Run python `./scripts/t-sne.py` to get the t-SNE visualization of the model's embeddings.
2. Run python `./scripts/t-SNE_all.py` to get the t-SNE visualization of the model's embeddings for all languages.

## Dataset

https://huggingface.co/datasets/badrex/nnti-dataset-full

## Contact maintainers

- [Dmitrii Tsybulkin](dmts00001@stud.uni-saarland.de)
- [Nicholas Pham](niph00001@stud.uni-saarland.de)
- [Regina Nitsch](reni00001@stud.uni-saarland.de)