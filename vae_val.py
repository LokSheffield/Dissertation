#!/usr/bin/python3

#Validates the VAE based confidence score, and compares with CNNs and Ensemble model with Distance based Loss & Softmax score
#dumps the figures into ./Figures directory

from defines import *
from utils import *

if not os.path.exists('Figures'):
    print("Note: Creating Figures directory")
    os.makedirs('Figures')

random.seed(VAE_SEED)
seed(VAE_SEED)
tf.random.set_seed(VAE_SEED)

#CNN Model instance
model_cnn = cnn_model(input_shape)

#Vector to store images for plotting
images_vec = []
images_vec.append(xtest)

#Vector of Acc values for plotting, including Ensemble model and test data
acc_vec_2d = [[0 for i in range(N_VAR+1)] for j in range(N_MODELS+1)]

#Smax+conf probability vector for Ensemble model
smax_vec = np.zeros((N_VAR+1,ytest.shape[0],N_MODELS))
#Class prediction vector for Ensemble model
clss_vec = (np.zeros((N_VAR+1,ytest.shape[0],N_MODELS))).astype(int)

#####################################################################
#VAE Model
#####################################################################
#Load the VAE Predictor Model Weights
if(DATA_SET == 'mnist'):
    enc_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(ENC_SEED)+'_VAE_model_mnist'
elif(DATA_SET == 'kmnist'):
    enc_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(ENC_SEED)+'_VAE_model_kmnist'
else:
    enc_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(ENC_SEED)+'_VAE_model_fashion_mnist'
enc_model = enc_model + '_KL_LOSS'
enc_model = enc_model+'.h5'
print("Model Name: ", enc_model)
#VAE Predictor Model instance
encoder, decoder, vae = vae_model(input_shape)
vae.summary()
#Loading weights
vae.load_weights('./RESULTS/'+enc_model)    

#Load the VAE Generator Model Weights
if(DATA_SET == 'mnist'):
    dec_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(DEC_SEED)+'_VAE_model_mnist'
elif(DATA_SET == 'kmnist'):
    dec_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(DEC_SEED)+'_VAE_model_kmnist'
else:
    dec_model = 'Best_Dims_'+str(LATENT_DIM)+'_Classes_'+str(N_CLASSES)+'_SEED_'+str(DEC_SEED)+'_VAE_model_fashion_mnist'
dec_model = dec_model + '_KL_LOSS'
dec_model = dec_model+'.h5'
#VAE Generator Model instance
encoder_gen, decoder_gen, vae_gen = vae_model(input_shape)
#Loading weights
vae_gen.load_weights('./RESULTS/'+dec_model)
mu_gen, sig_gen, _ = encoder_gen.predict(xtest)

#KNN Model Training
#Train & Test Embeddings
mu_train, _, _ = encoder.predict(xtrain)
#KNN Fit for Train Embeddings
knn = KNeighborsClassifier(n_neighbors=KNUM)
model_knn = knn.fit(mu_train, ytrain)

#Generate AUC Plot Metrics
fig, ax = plt.subplots(3,2, constrained_layout=True)
fig.suptitle("Data Set: "+DATA_SET, fontsize=12)
xx=[sigma+str(0),sigma+str(np.round(DELTA*1,2)),sigma+str(np.round(DELTA*2,2)),sigma+str(np.round(DELTA*3,2)),sigma+str(np.round(DELTA*4)),sigma+str(np.round(DELTA*5,2))]

#Loop across different CNN Models and validate the model for its Psmax vs Pvae from VAE
for cnn_seed in range(N_MODELS):
    #CNN weights
    if(DATA_SET == 'mnist'):
        cnn_wgts = './RESULTS/Best_DLoss_Model_mnist_'+str(cnn_seed)+'.h5'
    elif(DATA_SET == 'kmnist'):
        cnn_wgts = './RESULTS/Best_DLoss_Model_kmnist_'+str(cnn_seed)+'.h5'
    else:
        cnn_wgts = './RESULTS/Best_DLoss_Model_fashion_mnist_'+str(cnn_seed)+'.h5'

    #Load CNN Model weights
    model_cnn.load_weights(cnn_wgts)

    #Vectors to store auc scores for plotting
    smax_auc_vec = [] #Psmax auc
    kprb_auc_vec = [] #Pvae auc
    both_auc_vec = [] #Pcomb auc

    #####################################################################
    # Validation
    #####################################################################

    #Find the Softmax Probability, Confidence score and prediction(True/False)
    #form the CNN model
    cnn_prob, cnn_conf, cnn_pred, cnn_class = cnn_prob_conf_calc(model_cnn, xtrain, xtest, ytrain)

    #Find Accuracy vector
    acc = np.round(np.sum(cnn_pred)/ytest.shape[0],2)
    acc_vec_2d[cnn_seed][0] = acc

    #For Ensemble Model
    smax_vec[0,:,cnn_seed] = (cnn_prob+cnn_conf)/2
    clss_vec[0,:,cnn_seed] = cnn_class

    #####################################################################
    #Generating the Data set
    #####################################################################
    for ss in range(N_VAR):
        dev = ss*DELTA

        #Generate New Samples with increasing sigma values
        decout = decoder_gen.predict(sample_z([mu_gen,(sig_gen+dev)]))

        #Generate the mean(Latent) Vectors and Probabilities based on KNN of Training samples
        mux, _, _ = encoder.predict(decout)
        #Reshape the Generated samples
        xgen = decout.reshape(decout.shape[0], xtrain.shape[1], xtrain.shape[2])

        #Save Generated image to plot
        if(cnn_seed == 0):
            images_vec.append(xgen)

        #Check: Print Mu to make sure the Generating and Validating VAE models are different
        # mux_0, _, _ = encoder_gen.predict(decout)
        # print("Mu Gen:", mux_0[0,:])
        # print("Mu:", mux[0,:])

        # Probability based on KNNs
        # Find the KNN labels for mu(Latent space of Test Data)
        # from the Train Data set Latent space.
        # Calculte the probability based on Mode(most probable estimate of Y)
        # of the KNN samples
        # Probability = Mode of KNN samples/KNN
        knn_prob = knn_prob_calc(model_knn, mux, ytrain)

        #Find the CNN predictions and probabilities for Generated samples
        cnn_prob, cnn_conf, cnn_pred, cnn_class = cnn_prob_conf_calc(model_cnn, xtrain, xgen, ytrain)

        #Ensemble 
        smax_vec[ss+1,:,cnn_seed] = (cnn_prob+cnn_conf)/2
        clss_vec[ss+1,:,cnn_seed] = cnn_class
        
        #Accuracy
        acc = np.round(np.sum(cnn_pred)/ytest.shape[0],2)
        acc_vec_2d[cnn_seed][ss+1] = acc

        # smax_auc = np.round(roc_auc_score(cnn_pred,cnn_prob),3)
        smax_auc = np.round(roc_auc_score(cnn_pred,(cnn_prob+cnn_conf)/2),3)
        smax_auc_vec.append(smax_auc)
        kprb_auc = np.round(roc_auc_score(cnn_pred,knn_prob),3)
        kprb_auc_vec.append(kprb_auc)
        # both_auc = np.round(roc_auc_score(cnn_pred,(knn_prob+cnn_prob)/2),3)
        both_auc = np.round(roc_auc_score(cnn_pred,(knn_prob+cnn_prob+cnn_conf)/3),3)
        both_auc_vec.append(both_auc)

    ax[int(cnn_seed/2),(cnn_seed%2)].plot(xx,smax_auc_vec, color='b', label='CNN Pcomb')
    ax[int(cnn_seed/2),(cnn_seed%2)].plot(xx,kprb_auc_vec, color='g', label='VAE Pvae')
    ax[int(cnn_seed/2),(cnn_seed%2)].plot(xx,both_auc_vec, color='orange', label='Pcomb+Pvae')
    ax[int(cnn_seed/2),(cnn_seed%2)].legend(fontsize=7)
    ax[int(cnn_seed/2),(cnn_seed%2)].tick_params(axis='x', labelsize=7)
    ax[int(cnn_seed/2),(cnn_seed%2)].tick_params(axis='y', labelsize=7)    
    ax[int(cnn_seed/2),(cnn_seed%2)].set_title('Model-'+str(cnn_seed), fontsize=9)
    if((cnn_seed%2) == 0):
        ax[int(cnn_seed/2),(cnn_seed%2)].set_ylabel("AUC", fontsize=9)
    
plt.savefig('./Figures/'+DATA_SET+'_CNN_VAE_AUC_Metrics.png')
plt.show()

# Plot the Test & Generated Images
figp, axp = plt.subplots(N_VAR+1,5)
figp.suptitle("Data Set: "+DATA_SET, fontsize=12)
icount = 0
for image in images_vec:
    plt_idx = 0
    for ii in range(10):
        if((DATA_SET == 'mnist' and (ii == 0 or ii == 5 or ii == 3 or ii == 8 or ii == 9)) or
           (DATA_SET == 'kmnist' and (ii == 0 or ii == 2 or ii == 5 or ii == 7 or ii == 8)) or
           (DATA_SET == 'fashion_mnist' and (ii == 1 or ii == 2 or ii == 4 or ii == 6 or ii == 9))):
            axp[icount,plt_idx].imshow(image[list(ytest).index(ii)], cmap=pyplot.get_cmap('gray'))
            if(icount == 0):
                axp[icount,plt_idx].set_title('class: '+str(ii),fontsize=8)
            if(plt_idx == 0 and icount == 0):
                axp[icount,plt_idx].set_ylabel('xtest',fontsize=8)
            elif(plt_idx == 0):
                axp[icount,plt_idx].set_ylabel(xx[icount-1],fontsize=8)
            axp[icount,plt_idx].set_xticks([])
            axp[icount,plt_idx].set_yticks([])
            plt_idx = plt_idx+1
    icount = icount+1
figp.savefig('./Figures/'+DATA_SET+'_Images_sigma'+'.png')
plt.show()

#Ensemble Model
fige, axe = plt.subplots(int(N_VAR/3),3)
fige.suptitle("Data Set: "+DATA_SET, fontsize=12)
for ii in range(N_VAR+1):
    clss_vec1 = clss_vec[ii,:,:]
    smax_vec1 = smax_vec[ii,:,:]
    #Find the Mode of Class vector
    modey = np.array([np.bincount(element).argmax() for element in clss_vec1])
    modey_rep = np.repeat(np.reshape(modey,[modey.shape[0],1]),N_MODELS,axis=1)
    mask = (modey_rep == clss_vec1)*1
    #Find the Average Smax(probability)
    en_prob = np.sum(smax_vec1*mask,axis=1)/np.sum(mask,axis=1)
    #Find the True/False prediction
    en_pred = (np.equal(modey,ytest)*1)

    #Accuracy of the Ensemble model
    acc = np.round(np.sum(en_pred)/ytest.shape[0],2)
    acc_vec_2d[-1][ii] = acc
    #AUC for Ensemble
    if(ii > 0):
        auc1 = str(np.round(roc_auc_score(en_pred,en_prob),3))
        auc3 = str(np.round(roc_auc_score(en_pred,knn_prob),3))
        auc4 = str(np.round(roc_auc_score(en_pred,(knn_prob+en_prob)/2),3))
        fpr1, tpr1, _ = roc_curve(en_pred,en_prob)
        fpr3, tpr3, _ = roc_curve(en_pred,knn_prob)
        fpr4, tpr4, _ = roc_curve(en_pred,(knn_prob+en_prob)/2)
        axe[int((ii-1)/3),(ii-1)%3].plot(fpr1,tpr1, label = 'Ensemble, auc:'+auc1)
        axe[int((ii-1)/3),(ii-1)%3].plot(fpr3,tpr3, label = 'VAE, auc:'+auc3)
        axe[int((ii-1)/3),(ii-1)%3].plot(fpr4,tpr4, label = 'VAE+Ensemble, auc:'+auc4)
        axe[int((ii-1)/3),(ii-1)%3].legend(fontsize=7)
        axe[int((ii-1)/3),(ii-1)%3].set_title(xx[(ii-1)],fontsize=8)
        axe[int((ii-1)/3),(ii-1)%3].tick_params(axis='x', labelsize=7)
        axe[int((ii-1)/3),(ii-1)%3].tick_params(axis='y', labelsize=7)    
        if((ii-1)%3 == 0):
            axe[int((ii-1)/3),(ii-1)%3].set_ylabel('True Positive Rate', fontsize=8)
        if(int((ii-1)/3) == 1):
            axe[int((ii-1)/3),(ii-1)%3].set_xlabel('False Positive Rate', fontsize=8)
plt.savefig('./Figures/'+DATA_SET+'_Ensemble_ROC'+'.png')
plt.show()

#Plot Accuracy for all models
for ii in range(N_MODELS+1):
    if(ii == N_MODELS):
        plt.plot(['xtest']+xx,acc_vec_2d[ii],label='Model: Ensemble')
    else:
        plt.plot(['xtest']+xx,acc_vec_2d[ii],label='Model: '+str(ii))
    plt.ylabel('Accuracy',fontsize=11)
    plt.xlabel('Test and Generated data samples with varying Sigma',fontsize=11)
    plt.legend(fontsize=8)
    plt.title("Data Set: "+DATA_SET, fontsize=12)
plt.savefig('./Figures/'+DATA_SET+'_Accuracy_sigma'+'.png')
plt.show()
    
