from pathlib import Path
import torch
import pathlib
import torch

def save_model(target_directory: pathlib.Path,
               model: torch.nn.Module,
               model_name: str):
  assert model_name[-4:] == ".pth" or model_name[-3:] == ".pt", "Remember about .pth or .pt at the end of model_name!"
  if not target_directory.is_dir():
    target_directory.mkdir(parents=True,
                           exist_ok=True)
  # Save model weights to target directory
  print(f"Saving model to: {target_directory}")
  torch.save(model.state_dict(), Path(target_directory) / model_name)
