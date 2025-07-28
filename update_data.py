import os
import shutil
import tarfile
import tempfile
import requests
from tqdm import tqdm

# 常量路径配置
DOWNLOAD_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
TARGET_DIR = r"D:\gitdesktop\Qtrade\cn_data"
DOWNLOAD_PATH = os.path.join(tempfile.gettempdir(), "qlib_bin.tar.gz")

# 要替换的三个子目录
REPLACE_DIRS = ["calendars", "features", "instruments"]

def download_file(url, output_path):
    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    print(f"Downloaded to {output_path}")

def extract_and_replace(tar_path, target_dir):
    with tempfile.TemporaryDirectory() as tmp_extract_dir:
        print(f"Extracting {tar_path} to {tmp_extract_dir} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=tmp_extract_dir)
        print("Extraction complete.")

        # 自动检测根目录
        root_dirs = [d for d in os.listdir(tmp_extract_dir) if os.path.isdir(os.path.join(tmp_extract_dir, d))]
        if len(root_dirs) != 1:
            raise RuntimeError(f"Unexpected archive structure: found {len(root_dirs)} top-level dirs")
        qlib_root = os.path.join(tmp_extract_dir, root_dirs[0])

        for subdir in REPLACE_DIRS:
            src = os.path.join(qlib_root, subdir)
            dst = os.path.join(target_dir, subdir)
            if os.path.exists(dst):
                print(f"Removing old directory: {dst}")
                shutil.rmtree(dst)
            print(f"Copying {src} to {dst}")
            shutil.copytree(src, dst)
        print("Replacement complete.")

def main():
    try:
        download_file(DOWNLOAD_URL, DOWNLOAD_PATH)
        extract_and_replace(DOWNLOAD_PATH, TARGET_DIR)
        print("✅ All done.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
