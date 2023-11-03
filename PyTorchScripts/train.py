"""
Train a PyTorch model using device agnostic code.
"""
import os
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, train, utils # This will we in the same directory

import argparse

# Specify the parser
parser = argparse.ArgumentParser(description="Specify the most important hymerparameters when training multi-class classification model.")

# Get an arguments for training and test data
parser.add_argument("--training_data_path", type=str, help="Specify the training data path", required=True)
parser.add_argument("--testing_data_path", type=str, help="Specify the testing data path", required=True)

# Get an argument for number of epochs
parser.add_argument("--n_epochs", type=int, help="Specify the number of epochs to train for", default=10)

# Get an argument for the batch_size
parser.add_argument("--batch_size", type=int, help="Specify the size of the batches", default=32)

# Get an argument for learning_rate
parser.add_argument("--learning_rate", type=float, help="Specify the learning_rate for our optimizer", default=0.001)

# Get an argument for number of hidden units in TinyVGG model
parser.add_argument("--hidden_units", type=int, default=8, 
                    help="Specify the number of hidden_units that we are going to use inside our model's architecture")
# Get the above arguments
args = parser.parse_args()
print(f"Arguments: {args}")

# Setup some hyperparameters
INPUT_SHAPE = 3 # Number of color channels
OUTPUT_SHAPE = 3 # We have 3 possible classes
HIDDEN_UNITS = args.hidden_units
NUM_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

# Setup directories
train_dir = args.training_data_path
test_dir = args.testing_data_path

print(f"""Training the model with following parameters:
      Train data path: {train_dir} | Testing data path: {test_dir}
      Batch size: {BATCH_SIZE} | Number of epochs: {NUM_EPOCHS}
      Learning rate: {LEARNING_RATE} | Hidden units: {HIDDEN_UNITS}""")

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=[64, 64]),
    # Here we can of course add soe data augmentation if we want to add diversity to our training data
    transforms.ToTensor()
])

# Get the preprocessed data
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                               test_dir,
                                                                               data_transform,
                                                                               batch_size=BATCH_SIZE)
# Create the model
model = model_builder.TinyVGG(INPUT_SHAPE, HIDDEN_UNITS, OUTPUT_SHAPE).to(device)

# Setup the loss and the optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

start = timer()
# Start training with help from engine.py script
engine.train(model, train_dataloader, test_dataloader,
             optimizer, loss, NUM_EPOCHS, device)
end = timer()
print(f"It took {end - start} seconds to train he model.")

# Save the model to file
utils.save_model(model,
                 target_dir="models",
                 model_name="VGG_model_1.pth") # Could also by .py for model_name
