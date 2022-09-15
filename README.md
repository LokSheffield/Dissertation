# Dissertation

Files:
---------------------------------------------------------------------------
defines.py   - Global variables defined and python packages required for the project\
utils.py     - All Function definitions for the project\
cnn_train.py - File to train the CNN Models\
vae_train.py - File to train the VAE Models\
cnn_val.py   - Validates the CNN models, the regular one and Distance based Loss one\
vae_val.py   - Validates the VAE based confidence score, and compares with CNNs and Ensemble model with Distance based Loss & Softmax score\

Folders:
---------------------------------------------------------------------------
RESULTS - Contains all the model parameter files and CNN training results text files\
DATA    - kmnist data set files\
Figures - Figures used by the project document\
\
Note: These folders will be created after running the python files\

Commands: 
---------------------------------------------------------------------------
Note:\
-Before running any file, the desired data set variable(DATA_SET) should be set in the defines.py.\
-The kmnist data has to be downloaded before you start any runs for this dataset\
\
Download Kmnit dataset : python download_data.py\
Train CNN              : python cnn_train.py\
Train VAE              : python vae_train.py (run twice by setting VAE_SEED to 0 and then 1 in defines.py)\
Validate CNN           : python cnn_val.py (dumps the Figures to ./Figures directory)\
Validate VAE           : python vae_val.py (dumps the Figures to ./Figures directory)\

Tool/Package Versions:
---------------------------------------------------------------------------
Tensor Flow : 2.8.2\
Pandas      : 1.3.5\
Numpy       : 1.21.6\
Keras       : 2.8.0\
Python3     : 3.7.6\
