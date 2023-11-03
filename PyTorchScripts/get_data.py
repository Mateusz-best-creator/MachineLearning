"""
Using this script you can download the data from the external resource.
"""

import os
import zipfile

from pathlib import Path
import pathlib

import requests

def download_data(url: str,
                  data_directory: str):
  """
  Get the .zip file from the github, extract data from this file.
  """
  # Get the name of the .zip file
  zip_filename = url.split("/")[-1]
  zip_data_name = zip_filename[:-4]

  if (Path(data_directory) / "test").is_dir() and (Path(data_directory) / "train").is_dir():
    print(f"Data is arleady downloaded.")
  else:
    Path(data_directory).mkdir(parents=True, exist_ok=True)
    request = requests.get(url)
    request.raise_for_status()

    # Write the zip file to our data directory
    print(f"Getting {zip_filename}, saving to: {Path(data_directory) / zip_filename}.")
    with open(Path(data_directory) / zip_filename, "wb") as f:
      f.write(request.content)

    # Unzip the file
    zip_directory = Path(data_directory) / zip_filename
    print(f"Extracting data from {zip_filename} to {data_directory}.")
    with zipfile.ZipFile(zip_directory, "r") as zip_ref:
      zip_ref.extractall(Path(data_directory))

    # After everything is done we can remove .zip file'
    print(f"Removing zip file...")
    os.remove(Path(data_directory) / zip_filename)


def download_helper_scripts(git_repository_name: str,
                            target_directory: str="scripts"):
  """
  Downloads helper scripts from github.
  """
  data_path = Path(target_directory) / git_repository_name
  if data_path.is_dir():
    print(f"Scripts are arleady downloaded, skipping download...")
  else:
    print(f"Creating directory: {data_path}")
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data from: https://github.com/Mateusz-best-creator/PyTorch_Scripts")
    !git clone https://github.com/Mateusz-best-creator/PyTorch_Scripts
    !mv PyTorch_Scripts scripts
