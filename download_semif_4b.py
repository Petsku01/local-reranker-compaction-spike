from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    cache_dir="<spike_dir>/hf-cache",
)
print(path, flush=True)
