#!/usr/bin/python3

#Trains the CNN for N_MODELS and dumps the weights into ./RESULTS directory

from defines import *
from utils import *

if not os.path.exists('RESULTS'):
    print("Note: Creating Results directory")
    os.makedirs('RESULTS')

#Train the Model for N_MODELS
for ii in range(N_MODELS):
    SEED = ii
    MODEL_NAME_int = MODEL_NAME+'_'+str(SEED)+'.h5'

    print("################################ SEED: ",SEED)
    random.seed(SEED)
    seed(SEED)
    tf.random.set_seed(SEED)

    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)#, restore_best_weights=True)
    check_point_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_NAME_int, save_best_only=True, verbose=True)

    input_shape = [xtest.shape[1],xtest.shape[2],1]
    model = cnn_model(input_shape)

    history = model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(xtest,ytest), verbose=2, shuffle=True, callbacks=[check_point_cb, early_stop_cb, CustomCallback()])

    #Obtain Feature Vectors from the Penultimate Layer of the Model
    y_train_fvecs = model.predict(xtrain)

    #Predictions on the test set
    y_pred_fvecs = model.predict(xtest)
    y_pred_smax = Softmax()(y_pred_fvecs)
    y_pred_class = np.array([np.argmax(element) for element in y_pred_smax])
    ytest = (tf.squeeze(ytest))
    ypred = tf.constant(y_pred_class)
    cm = confusion_matrix(ytest, ypred)
    print("Confusion Matrix: Test set:\n", cm)
    mask = np.equal(ypred,ytest)*1
    acc = np.sum(mask)/ytest.shape[0]
    print("Accuracy: Test set: ", acc)
    
    #LDA Analysis to validate the Embeddings in the Penultimate layer(Latent space)
    #Training Fvecs LDA
    lda_trn = LDA()
    lda_trn.fit_transform(y_train_fvecs, ytrain)
    print("Train data LDA :", lda_trn.explained_variance_ratio_)
    #Testing Fvecs LDA
    lda_tst = LDA()
    lda_tst.fit_transform(y_pred_fvecs, ytest)
    print("Test data LDA :", lda_tst.explained_variance_ratio_)

    #Store the Results to a file
    with open(OUT_FILE, "a") as ofile:
        print("SEED: ", SEED, file=ofile)
        print("Test Acc: ", acc, file=ofile)
        print("Confusion Matrix: \n",cm, file=ofile)
        print("LDA Trained: ", lda_trn.explained_variance_ratio_, file=ofile)
        print("LDA Test: ", lda_tst.explained_variance_ratio_, file=ofile)
        ofile.close()
    #move the model to Results Directory
    os.system("mv %s %s"%(MODEL_NAME_int, './RESULTS/'))
        
os.system("mv %s %s"%(OUT_FILE, './RESULTS/'))
