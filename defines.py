#File Load necessary Python packages and includes Global variables for the project
#Load Packages
#For ML
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Reshape, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, Softmax, BatchNormalization
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import keras.backend as KB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.losses import binary_crossentropy
from yellowbrick.classifier.rocauc import roc_auc
from sklearn import decomposition
from sklearn.preprocessing import scale
from scipy import stats
import numpy as np
import pandas as pd
#For Plots
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from plotly.offline import init_notebook_mode, iplot
import matplotlib as mpl
import matplotlib
from keras.utils.vis_utils import plot_model
#Misc
import itertools
from operator import add
import os
import time
import platform
import sys
import random
from numpy.random import seed

#Set Plot style
plt.style.use('ggplot')

#Package versions
if(str(tf.__version__) != '2.8.2'):
    print("*****WARNING*****")
    print("Use Tensorflow Version: 2.8.2")
if(str(pd.__version__) != '1.3.5'):
    print("*****WARNING*****")
    print("Use Pandas Version: 1.3.5")
if(str(np.__version__) != '1.21.6'):
    print("*****WARNING*****")
    print("Use Pandas Version: 1.21.6")
if(str(tf.keras.__version__) != '2.8.0'):
    print("*****WARNING*****")
    print("Use Pandas Version: 2.8.0")
if(platform.python_version() != '3.7.6'):
    print("*****WARNING*****")
    print("Use Python3 Version: 3.7.6")
    
#Selects the Data set
#Allowed data sets are fashion_mnist, kmnist & mnist
DATA_SET = 'fashion_mnist'

#When set 1, Enables Distance Loss function during CNN Training
EN_DLOSS = 0

#Repeat Training for N_MODELS, each iteration dumps a new parameter set for the Model
N_MODELS = 6

#Hyper Parameters for Training
#Max Number of Epochs allowed
N_EPOCHS = 75
#Training Batch size
BATCH_SIZE = 100
#Stop the Training once Max Accuracy is reached
MAX_ACCURACY = 0.99
#Stop training if the results doesn't improve by a number of Epochs defined by PATIENCE
PATIENCE = 25
#K Nearest Neighbours for KNN model training
KNUM = 50

if(EN_DLOSS):
    MODEL_NAME = "Best_DLoss_Model"
    OUT_FILE = "Results_DLoss"
else:
    MODEL_NAME = "Best_No_DLoss_Model"
    OUT_FILE = "Results_No_DLoss"
    
#Import Data Sets
if(DATA_SET == 'mnist'):
    from tensorflow.keras.datasets import mnist
    (xtrain,ytrain), (xtest,ytest) = mnist.load_data()
    MODEL_NAME = MODEL_NAME + '_mnist'
    OUT_FILE = OUT_FILE + '_mnist'
elif(DATA_SET == 'kmnist'):
    def load(f):
        return np.load(f)['arr_0']
    xtrain = load('./DATA/kmnist-train-imgs.npz')
    xtest  = load('./DATA/kmnist-test-imgs.npz')
    ytrain = load('./DATA/kmnist-train-labels.npz')
    ytest  = load('./DATA/kmnist-test-labels.npz')
    MODEL_NAME = MODEL_NAME + '_kmnist'
    OUT_FILE = OUT_FILE + '_kmnist'
else:
    from tensorflow.keras.datasets import fashion_mnist
    (xtrain,ytrain), (xtest,ytest) = fashion_mnist.load_data()
    MODEL_NAME = MODEL_NAME + '_fashion_mnist'
    OUT_FILE = OUT_FILE + '_fashion_mnist'

#Make sure the values of input is float and range from 0 to 1
xtrain = xtrain.astype('float32')/xtrain.max()
xtest = xtest.astype('float32')/xtest.max()

#Number of classes in the data set
N_CLASSES = len(np.unique(ytest))
    
#Latent Dimensions for VAE
LATENT_DIM = N_CLASSES

#Train VAE on two different seeds, one for Predictor(VAE_SEED = 0)
#and other for Generator(VAE_SEED = 1)
VAE_SEED = 0

#Variable to vary the steps of sigma to generate the novel data
DELTA = 0.25

#Number of steps the sigma can be changed
N_VAR = 6

#Don't Change these..
#File to store the Training results for N_MODELS models
OUT_FILE = OUT_FILE+'.txt'

#VAE Predictor Model seed
ENC_SEED = 0
#VAE Generator Model seed
DEC_SEED = 1

#To print sigma greek letter
sigma = '\u03C3+'

#Disable the GPU execution
#Set this always to 1 to reproduce the results for a SEED setting
DISABLE_GPU = 1

#Disables the GPU
if(DISABLE_GPU):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Color map used    
CMAP = matplotlib.cm.get_cmap('viridis')

#Input shape for CNN and VAE models
input_shape = [xtest.shape[1],xtest.shape[2],1]
