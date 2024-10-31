import os
from pathlib import Path

def create_symlinks(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Ensure the destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    for file in source_path.iterdir():
        if file.is_file():
            # Define the symlink path in the destination directory
            symlink_path = dest_path.joinpath(file.name)
            
            # Create the symlink
            try:
                os.symlink(file, symlink_path)
                print(f"Created symlink: {symlink_path} -> {file}")
            except FileExistsError:
                print(f"Symlink already exists: {symlink_path}")
            except OSError as e:
                print(f"Error creating symlink: {symlink_path} -> {file}, {e}")

# Example usage
source_dir = '/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_liv_preprocessed/'  # Replace with the actual source directory path
dest_dir = '/mnt/qdata/rawdata/UKBIOBANK/ukb_70k/interim/ukb_liv_preprocessed/'  # Replace with the actual destination directory path
create_symlinks(source_dir, dest_dir)