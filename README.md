# MarioVL

Make not that this code directly makes use of  of the MarioNette code base. The MarioNette project page can be found [here](https://people.csail.mit.edu/smirnov/marionette/) and the corresponding github page can be found [here](https://github.com/dmsm/MarioNette).

## Setup

Please install the required packages using the requirements.txt file

## Data

The dataset can be found with the Google Drive link [here](https://drive.google.com/open?id=1vzFVFhJZDZMkJ8liROtIyzOiUY42r4TZ).

You can download the dataset and place it in the data folder. The data folder should be in the same directory as the code.

To prepare the dataset, you can run the prep_data.py file. Note that the code assumes you have the image dataset in the data/images.

## Training

To train the model, you can run the scripts/train.py file. You will need to provde a --data argument with the path to the created dataset as well as the --checkpoint_dir argument with the path to the directory where you want to save the checkpoints.