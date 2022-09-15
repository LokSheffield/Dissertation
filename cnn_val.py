#!/usr/bin/python3

#Validates the CNN models, the regular one and Distance based Loss one
#dumps the figures into ./Figures directory

from defines import *
from utils import *

if not os.path.exists('Figures'):
    print("Note: Creating Figures directory")
    os.makedirs('Figures')

#DLoss and No DLoss models
MODEL1 = "Best_DLoss_Model_"+DATA_SET
MODEL2 = "Best_No_DLoss_Model_"+DATA_SET

#Variables for Plotting
#Area Under the curve Vector - stores the AUC metrics for different SEED
auc_vec_m1 = []
auc_vec_m2 = []
auc_vec_mix_m1 = []
auc_vec_mix_m2 = []
#Accuracy Vector - stores the Accuracy of the model for different SEED
acc_vec_m1 = []
acc_vec_m2 = []
#LDA Vector - stores the Accuracy of the model for different SEED
lda_vec_m1 = []
lda_vec_m2 = []
#To save M1 models Predictions and Conf scores
conf_vec_m1 = []
pred_vec_m1 = []

CNN_SEED = 0
random.seed(CNN_SEED)
seed(CNN_SEED)
tf.random.set_seed(CNN_SEED)

#CNN Model instance
model = cnn_model(input_shape)

# plot_model(model, to_file='./Figures/CNN_Model_graph.png', show_shapes=True, show_layer_names=True)

#Figure to ploat latent spaces
fig = plt.figure(figsize=(20,10))

#Load the Model for N_MODELS(6)
for ii in range(N_MODELS):
    SEED = ii
    MODEL1_int = MODEL1+'_'+str(SEED)+'.h5'
    MODEL2_int = MODEL2+'_'+str(SEED)+'.h5'

    for jj in range(2):
        if(jj == 0):#Load Model1 weights
            model.load_weights('./RESULTS/'+MODEL1_int)
        else:#Load Model2 weights
            model.load_weights('./RESULTS/'+MODEL2_int)

        #Obtain Feature Vectors from the Penultimate Layer of the Model for Train Set
        y_train_fvecs = model.predict(xtrain)
        #Obtain Feature Vectors from the Penultimate Layer of the Model for Test Set
        y_pred_fvecs = model.predict(xtest)

        #Call the function to return the results
        cnn_prob, cnn_conf, cnn_pred, cnn_class = cnn_prob_conf_calc(model, xtrain, xtest, ytrain)

        #Find Accuracy
        acc = np.round(np.sum(cnn_pred)/ytest.shape[0],2)
        #For plotting
        if(jj == 0):
            acc_vec_m1.append(acc)
        else:
            acc_vec_m2.append(acc)

        #AUC score
        if(jj == 0):
            #AUC for M1 model with confidence score
            auc = np.round(roc_auc_score(cnn_pred,cnn_conf),2)
            auc_vec_m1.append(auc)
            #Save M1 conf score
            conf_vec_m1.append(cnn_conf)
            #Save M1 predictions
            pred_vec_m1.append(cnn_pred)
        else:
            #AUC for M2 model with Smax score
            auc = np.round(roc_auc_score(cnn_pred,cnn_prob),2)
            auc_vec_m2.append(auc)
            #AUC for M1 model with combined probabilities
            auc = np.round(roc_auc_score(pred_vec_m1[-1],(cnn_prob+conf_vec_m1[-1])/2),2)
            auc_vec_mix_m1.append(auc)
            #AUC for M2 model with combined probabilities
            auc = np.round(roc_auc_score(cnn_pred,(cnn_prob+conf_vec_m1[-1])/2),2)
            auc_vec_mix_m2.append(auc)
            
        #LDA Analysis to validate the Embeddings in the Penultimate layer(Latent space)
        #Training Fvecs LDA
        lda_trn = LDA()
        lda_trn.fit_transform(y_train_fvecs, ytrain)
        # print("Train data LDA :", lda_trn.explained_variance_ratio_)
        #Testing Fvecs LDA
        lda_tst = LDA()
        lda_tst.fit_transform(y_pred_fvecs, ytest)
        # print("Test data LDA :", lda_tst.explained_variance_ratio_)
        #For plotting
        if(jj == 0):
            lda_vec_m1.append(np.round(lda_tst.explained_variance_ratio_[0],2))
        else:
            lda_vec_m2.append(np.round(lda_tst.explained_variance_ratio_[0],2))

        #Apply PCA to reduce the dimensions for plots
        pca = decomposition.PCA(n_components=3)
        pcain = (y_pred_fvecs)
        scores = pca.fit_transform(pcain)
        #Make a DataFrame
        scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
        ytest_df = pd.DataFrame(ytest, columns=['Ytest'])
        df_scores = pd.concat([scores_df, ytest_df], axis=1)

        #Plot the latent space
        if(ii == 0):# Print for the first Model
            ax = fig.add_subplot(1,2,jj+1, projection = '3d')
            ax.scatter(df_scores['PC1'],df_scores['PC2'],df_scores['PC3'],c=CMAP(df_scores['Ytest']/9),alpha=0.7)
            if(jj == 0):
                ax.set_title('DLoss & Conf')
            else:
                ax.set_title('No DLoss & Smax')
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-5, 10)
plt.savefig('./Figures/'+DATA_SET+'_DlossvsNoDloss_Latent.png')
plt.show()

#Plot the color bar
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                               cmap=CMAP,
                               norm=mpl.colors.Normalize(0, 10),  # vmax and vmin
                               extend='both',
                               # label='This is a label',
                               ticks=[0,1,2,3,4,5,6,7,8,9])
cb.ax.tick_params(labelsize=30)
plt.savefig('./Figures/just_colorbar.png', bbox_inches='tight')
plt.show()

#Plot the Accuracy, LDA and AUC metrics for all the Models
#Model number for x axis
xx=[0,1,2,3,4,5]
#Figure for plotting Accuracy, LDA and AUC
fig, ax = plt.subplots(3,1, constrained_layout=True)
fig.suptitle("Data Set: "+DATA_SET, fontsize=12)

#Accuracy plot
print("Model1 Acc: ",acc_vec_m1)            
print("Model2 Acc: ",acc_vec_m2)            
ax[0].plot(xx,acc_vec_m1, color='b', label='DLoss & Conf')
ax[0].plot(xx,acc_vec_m2, color='g', label='no DLoss & Smax')
ax[0].legend(fontsize=9)
ax[0].set_ylabel("Accuracy", fontsize=10)
ax[0].tick_params(axis='x', labelsize=7)
ax[0].tick_params(axis='y', labelsize=7)    

#LDA plot
print("Model1 LDA: ",lda_vec_m1)            
print("Model2 LDA: ",lda_vec_m2)            
ax[1].plot(xx,lda_vec_m1, color='b', label='DLoss & Conf')
ax[1].plot(xx,lda_vec_m2, color='g', label='no DLoss & Smax')
ax[1].legend(fontsize=9)
ax[1].set_ylabel("LDA explained variance", fontsize=10)
ax[1].tick_params(axis='x', labelsize=7)
ax[1].tick_params(axis='y', labelsize=7)    

#AUC plot
print("Model1 AUC with conf: ",auc_vec_m1)            
print("Model2 AUC with smax: ",auc_vec_m2)            
print("Model1 AUC with conf+smax: ",auc_vec_mix_m1)            
print("Model2 AUC with conf+smax: ",auc_vec_mix_m2)            
ax[2].plot(xx,auc_vec_m1, color='b', label='DLoss & Conf')
ax[2].plot(xx,auc_vec_m2, color='g', label='no DLoss & Smax')
ax[2].plot(xx,auc_vec_mix_m1, color='orange', label='DLoss & Conf+Smax')
ax[2].plot(xx,auc_vec_mix_m2, color='gray', label='no DLoss & Conf+Smax')
ax[2].legend(fontsize=9)
ax[2].set_xlabel("Models", fontsize=10)
ax[2].set_ylabel("AUC score", fontsize=10)
ax[2].tick_params(axis='x', labelsize=7)
ax[2].tick_params(axis='y', labelsize=7)    

plt.savefig('./Figures/'+DATA_SET+'_DlossvsNoDloss_Metrics.png')
plt.show()           
           
