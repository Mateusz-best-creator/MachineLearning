# Going Modular Scripts

They breakdown as follows: 
* `data_setup.py` - a file to prepare and download data if needed.
* `save_model.py` - a file that contains a function to save our model.
* `engine.py` - a file containing various training functions.
* `model_builder.py` - a file to create a PyTorch TinyVGG model.
* `utils.py` - a file dedicated to helpful utility functions.
* `predictions.py` - a file for making predictions with a trained PyTorch model and input image (the main function, `pred_and_plot_image()`)
* `get_data.py` - a script that allows you to upload the data from the external resource (you have to pass `data_path` - tell this function where to store our data, `url` - specify from where we want to upload the data, `zip_filename` - specify .zip filename which you are going to download)
* `train.py` - a file to leverage all other files and train a target PyTorch model. You can also pass multiple hyperparameters when excuting this script. Some example how you can execute this script: `!python going_modular/train.py --training_data_path="data/pizza_steak_sushi/train" --testing_data_path="data/pizza_steak_sushi/test" --batch_size=128 --hidden_units=16`.
* `visualization.py` - contains a functions for visualization purposes.
