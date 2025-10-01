from huggingface_hub import snapshot_download
from pathlib import Path

OV_REPO_ID = "OpenVINO/stable-diffusion-v1-5-int8-ov"  # SD 1.5 INT8 
MODEL_DIR  = Path("models/sd15_int8_ov")

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=OV_REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8
    )
    xmls = list(MODEL_DIR.rglob("*.xml"))
    bins = list(MODEL_DIR.rglob("*.bin"))
    print(f"Downloaded -> {len(xmls)} xml; {len(bins)} bin")
    if len(bins) == 0:
        print("xoá thư mục và chạy lại")
    else:
        print("Model sẵn sàng:", MODEL_DIR.resolve())
