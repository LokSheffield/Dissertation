#!/usr/bin/python3


#Trains the VAE Model and dumps the weights into ./RESULTS directory
#Make sure it is run twice, once by seeting VAE_SEED to 0 in defines.py and then setting to 1

from defines import *
from utils import *
tf.compat.v1.disable_eager_execution()

if not os.path.exists('RESULTS'):
    print("Note: Creating RESULTS directory")
    os.makedirs('RESULTS')

#Hyper Parameters for Training
#Batch size for Training
BATCH_SIZE = 128
#Number of Epochs for Training
N_EPOCHS = 100
#Split the Data for Training and Validation
VALIDATION_SPLIT = 0.2

random.seed(VAE_SEED)
seed(VAE_SEED)
tf.random.set_seed(VAE_SEED)

#Define Model Name to dump
if(DATA_SET == 'mnist'):
    model_name = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(VAE_SEED)+'_VAE_model_mnist'
elif(DATA_SET == 'kmnist'):
    model_name = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(VAE_SEED)+'_VAE_model_kmnist'
else:
    model_name = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(VAE_SEED)+'_VAE_model_fashion_mnist'

model_name = model_name + '_KL_LOSS'

model_name = model_name+'.h5'
print("Model Name: ", model_name)

# Data & model configuration
img_width, img_height = xtrain.shape[1], xtrain.shape[2]

#Third dimension of the Data set(1=Monochrome images)
NUM_CHANNELS = 1

# Reshape data
xtrain = xtrain.reshape(xtrain.shape[0], img_height, img_width, NUM_CHANNELS)
xtest = xtest.reshape(xtest.shape[0], img_height, img_width, NUM_CHANNELS)

early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)#, restore_best_weights=True)
check_point_cb = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True, verbose=True)

input_shape = (img_height, img_width, NUM_CHANNELS)
encoder, decoder, vae = vae_model(input_shape)
vae.summary()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('sparse_categorical_accuracy')
        val_acc = logs.get('val_sparse_categorical_accuracy')
        KB.print_tensor(self.model.history)

#Train VAE
vae.fit(xtrain, xtrain, epochs = N_EPOCHS, batch_size = BATCH_SIZE, validation_split = VALIDATION_SPLIT, shuffle=True, callbacks=[check_point_cb, early_stop_cb, CustomCallback()])
os.system("mv %s %s"%(model_name, './RESULTS/'))
