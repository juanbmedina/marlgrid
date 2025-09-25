import os
import shutil
import pytz
from datetime import datetime

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete folder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def rename_and_move_result(output_dir, dest_root, exp_name):
    def get_local_timestamp():
        tz = pytz.timezone("America/Bogota")  # Colombia time
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    subfolders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    if len(subfolders) != 1:
        print(f"Expected 1 subfolder in {output_dir}, found {len(subfolders)}.")
        exit(1)

    original_path = os.path.join(output_dir, subfolders[0])
    timestamp = get_local_timestamp()
    new_folder_name = f"{exp_name}_{timestamp}"
    new_path = os.path.join(dest_root, new_folder_name)

    print(f"Renaming {original_path} → {new_path}")
    shutil.move(original_path, new_path)

def copy_config_file(src_filename, config_dir='path/to/your/config/dir'):
    # Define source and target paths
    src_path = os.path.join(config_dir, src_filename)
    dst_dir = '/opt/conda/lib/python3.8/site-packages/marllib/envs/base_env/config'
    dst_path = os.path.join(dst_dir, src_filename)

    # Check that the file exists
    if not os.path.exists(src_path):
        print(f"Source file does not exist: {src_path}")
        return

    # Create the target directory if it doesn't exist (optional)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # Copy the file
    try:
        shutil.copy2(src_path, dst_path)
        print(f"Copied '{src_filename}' to MARLlib config directory.")
    except Exception as e:
        print(f"Error copying file: {e}")