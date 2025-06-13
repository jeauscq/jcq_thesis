import os
import subprocess
from pathlib import Path

output_dir = str(Path(__file__).resolve().parents[1])+"/"+"datasets"
print(f"Output directory: {output_dir}")

def download_gdrive_folder(folder_id="1MMt8RSr0ET6QHFbv-PF56vyQbqlHBxLM", output_dir=output_dir):
    # Ensure gdown is installed
    try:
        import gdown
    except ImportError:
        raise ImportError("Please install 'gdown' first: pip install gdown")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the download URL
    print(f"Starting download from Google Drive folder: {folder_id}")

    # Use gdown to recursively download the folder contents
    command = ["gdown", "--folder", folder_id, "-O", output_dir]
    subprocess.run(command, check=True)

    print(f"Download completed. Files saved in: {output_dir}")

if __name__ == "__main__":
    download_gdrive_folder()
