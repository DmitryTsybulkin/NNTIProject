import os

from huggingface_hub import HfApi

repo_id = "radioduran/mHuBERT-knnvc-dataset"
folder_path = "./knnvc_cache/mHuBERT-147_5sec_knnvc"

api = HfApi(token=os.getenv('HF_TOKEN'))

print(f"Repository creation {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

print(f"Loading the folder {folder_path}...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="knnvc_cache",
    commit_message="Upload knnvc cache data (original and knnvc_0 arrows)"
)

print("Successfully uploaded.")