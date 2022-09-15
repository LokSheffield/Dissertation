#!/usr/bin/python3
#File includes necessary functions for the Project

from defines import *

UTIL_SEED = 0
random.seed(UTIL_SEED)
seed(UTIL_SEED)
tf.random.set_seed(UTIL_SEED)

#Generate combinations of pair_size of length size for DLoss function
def gen_pairs(size, pair_size):
    g =itertools.combinations(range(size),pair_size)
    alist = list(g)
    # random.shuffle(alist)
    return alist

#Distance based Loss Function
def distance_loss_fn(y_pen, y, loss_in):
    #Scaling factor for loss-hyperparam
    alpha = 0.1
    #set loss to 0
    loss_d = 0.0
    #Choose % of the pairs in random
    frac = 0.1
    #Max Distance to be maintained different Classes-hyperparam
    max_dist = 6.0
    #Generate pairs of batch size
    plist = gen_pairs(y.shape[0],2)
    plist = np.array(random.sample(plist, int(np.round(len(plist)*frac))))
    xx = tf.cast(tf.gather(y_pen, plist[:,0]), tf.float32)
    yy = tf.cast(tf.gather(y_pen, plist[:,1]), tf.float32)
    y0 = tf.gather(y, plist[:,0])
    y1 = tf.gather(y, plist[:,1])
    comp = tf.equal(y0, y1)
    comp1 = tf.cast(comp, tf.float32)
    comp2 = tf.cast(tf.logical_not(comp), tf.float32)
    # print(comp)
    dist_t = tf.norm(tf.subtract(xx,yy),axis=1, keepdims=True)
    dist_f = tf.maximum(0.0,(tf.subtract(max_dist, dist_t)))
    dist = tf.add(tf.multiply(comp1, tf.squeeze(dist_t)),tf.multiply(comp2, tf.squeeze(dist_f)))
    dist = tf.reduce_mean(dist)
    loss_d = dist*alpha
    return loss_d

#Callback to Stop training when a desired Accuracy is reached
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('sparse_categorical_accuracy')
        val_acc = logs.get('val_sparse_categorical_accuracy')
        # KB.print_tensor(self.model.history)
        # print("History:", self.model.history.logs)
        if(val_acc > MAX_ACCURACY) :
            print("\nReached desired Accuracy(Training) of:", train_acc)
            print("\nReached Accuracy(Validation) of:", val_acc)
            self.model.stop_training = True

#Custom Model to augment the given Loss Function with a Custom Loss
class CustomModel(tf.keras.Model):
    def train_step(self, data):
     # Unpack the data. Its structure depends on your model and
     # on what you pass to `fit()`.
        x, y = data
        # print("Shapes of X and Y to Model: ", x.shape, y.shape)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            #Penultimate Vector
            y_pen = y_pred
            #Models Softmax Layer
            y_pred = Softmax()(y_pred)
            # KB.print_tensor(y_pred, message = 'Softmax Layer Output: ')
            # # Compute the loss value *for each Batch*
            # # (the loss function is configured in `compile()`)
            loss_self = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss_dist = 0
            if(EN_DLOSS):
                #Distance based loss
                loss_dist = distance_loss_fn(y_pen, y, loss_self)
            loss = loss_self+loss_dist
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

#Distance based Confidence score function
def conf_calc(y_train_fvecs, y_pred_fvecs, ytrain):
    #Y prediction Probabilities
    y_pred = Softmax()(y_pred_fvecs)
    y_pred_classes = np.array([np.argmax(element) for element in y_pred])
    ypred = y_pred_classes
    # ytest = (tf.squeeze(ytest))
    # ypred = tf.constant(y_pred_classes)
    ##KNN Fit
    knn = KNeighborsClassifier(n_neighbors=KNUM)
    model_knn = knn.fit(y_train_fvecs, ytrain)
    #Identify 10NN in Ytrain_Fvecs for Ypred_Fvecs
    kneighs = model_knn.kneighbors(y_pred_fvecs)
    #Indexes of K(10) NN
    kindx = np.array(kneighs[1])
    #Class Labels of 10 NN
    yy = np.take(ytrain,kindx)
    yy_hat = np.repeat(np.reshape(y_pred_classes,[y_pred_classes.shape[0],1]),KNUM,axis=1)
    #Distance from Predicted vector to 10NN vectors of Training samples
    vec_dist = np.array(kneighs[0])
    ##Confidence Calculation
    exp_dist = np.exp(vec_dist*-1)
    #Denominator
    sum_expd = np.sum(exp_dist,axis=1)
    #Numerator
    mask = np.equal(yy,yy_hat)*1
    numr = np.sum(exp_dist*mask, axis=1)
    #Confidence Score
    conf = numr/sum_expd
    return conf

#KNN based Probability calculation
def knn_prob_calc(knn_model, mu, ytrain):
    #Identify KNNs(Train) for Test Embeddings
    kneighs = knn_model.kneighbors(mu)
    #Indexes of K(KNUM) NN
    kindx = np.array(kneighs[1])
    #Class Labels of KNUM NN
    yy = np.take(ytrain,kindx)
    #Find Mode of KNN labels
    yy_mode = np.array([np.bincount(element).argmax() for element in yy])
    #Repeat the Mode value for KNUM times
    yy_mode_rep = np.repeat(np.reshape(yy_mode,[yy_mode.shape[0],1]),KNUM,axis=1)
    #Find the probability
    pmask = np.equal(yy_mode_rep,yy)*1
    prob = np.sum(pmask, axis=1)/KNUM
    return prob

#Find the Softmax Probability, Confidence score and prediction(True/False)
#form the CNN model
def cnn_prob_conf_calc(cnn_model, xtrain, xtest, ytrain):
    #Predictions on the Train Data set
    y_train_fvecs = cnn_model.predict(xtrain)
    #Predictions on the Test Data set
    y_pred_fvecs = cnn_model.predict(xtest)
    #Y prediction
    y_pred = Softmax()(y_pred_fvecs)
    ypred = np.array([np.argmax(element) for element in y_pred])
    #Confidence score Calculation
    conf = conf_calc(y_train_fvecs, y_pred_fvecs, ytrain)
    #True/False Classification for ROC
    diffx = (np.equal(ypred,ytest)*1)
    #Softmax Probability
    prob_max = np.max(y_pred,axis=1)
    return prob_max, conf, diffx, ypred

# =================
# CNN Model
# =================
def cnn_model(input_shape):
    #CNN Model
    in_layer     = tf.keras.Input(shape=input_shape)
    #Convoluton/Pool/BatchNorm Layers
    conv_layer1  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(in_layer)
    pool_layer1  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_layer1)
    conv_layer2  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(pool_layer1)
    pool_layer2  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_layer2)
    drop_layer2  = BatchNormalization()(pool_layer2)
    conv_layer3  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(drop_layer2)
    pool_layer3  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_layer3)
    drop_layer3  = BatchNormalization()(pool_layer3)
    conv_layer4  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(drop_layer3)
    pool_layer4  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_layer4)
    drop_layer4  = BatchNormalization()(pool_layer4)
    conv_layer5  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(drop_layer4)
    # pool_layer5  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_layer5)
    drop_layer5  = BatchNormalization()(conv_layer5)
    #Dense/Drop Layers
    flat_layer   = Flatten()(drop_layer5)
    dense_layer6 = Dense(units = 128, activation='relu')(flat_layer)
    drop_layer6  = BatchNormalization()(dense_layer6)
    dense_layer7 = Dense(units = 64, activation='relu')(drop_layer6)
    drop_layer7  = BatchNormalization()(dense_layer7)
    dense_layer8 = Dense(units = 32, activation='relu')(drop_layer7)
    drop_layer8  = BatchNormalization()(dense_layer8)
    out_layer    = Dense(units=N_CLASSES)(drop_layer8)
    
    model = CustomModel(in_layer, out_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = KB.shape(mu)[0]
  dim       = KB.int_shape(mu)[1]
  eps       = KB.random_normal(shape=(batch, dim))
  return mu + KB.exp(sigma / 2) * eps

# Data & model configuration
img_width, img_height = xtrain.shape[1], xtrain.shape[2]

# =================
# VAE Model
# =================
def vae_model(input_shape):
    #Encoder Model
    in_e_layer     = Input(shape=input_shape)
    #Convoluton/Pool/Drop Layers
    conv_e_layer1  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(in_e_layer)
    pool_e_layer1  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_e_layer1)
    conv_e_layer2  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(pool_e_layer1)
    pool_e_layer2  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_e_layer2)
    drop_e_layer2  = BatchNormalization()(pool_e_layer2)
    conv_e_layer3  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(drop_e_layer2)
    pool_e_layer3  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_e_layer3)
    drop_e_layer3  = BatchNormalization()(pool_e_layer3)
    conv_e_layer4  = Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu')(drop_e_layer3)
    pool_e_layer4  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_e_layer4)
    drop_e_layer4  = BatchNormalization()(pool_e_layer4)
    conv_e_layer5  = Conv2D(filters=28, kernel_size=(3,3), padding='same',activation='relu')(drop_e_layer4)
    # pool_e_layer5  = MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(conv_e_layer5)
    drop_e_layer5  = BatchNormalization()(conv_e_layer5)
    #Dense/Drop Layers
    flat_e_layer   = Flatten()(drop_e_layer5)
    dense_e_layer6 = Dense(units = 128, activation='relu')(flat_e_layer)
    drop_e_layer6  = BatchNormalization()(dense_e_layer6)
    dense_e_layer7 = Dense(units = 64, activation='relu')(drop_e_layer6)
    drop_e_layer7  = BatchNormalization()(dense_e_layer7)
    dense_e_layer8 = Dense(units = 32, activation='relu')(drop_e_layer7)
    drop_e_layer8  = BatchNormalization()(dense_e_layer8)

    mu      = Dense(LATENT_DIM, name='latent_mu')(drop_e_layer8)
    sigma   = Dense(LATENT_DIM, name='latent_sigma')(drop_e_layer8)

    conv_shape = KB.int_shape(drop_e_layer5)

    # Use reparameterization trick
    z = Lambda(sample_z, output_shape=(LATENT_DIM, ), name='z')([mu, sigma])

    # Instantiate encoder
    encoder = Model(in_e_layer, [mu, sigma, z], name='encoder')
    # print(encoder.summary())

    # Decoder
    # decoder takes the latent vector as input
    decoder_input = Input(shape=(LATENT_DIM, ))
    dense_d_layer0 = Dense(32, activation='relu')(decoder_input)
    dense_d_layer01 = Dense(64, activation='relu')(dense_d_layer0)
    dense_d_layer02 = Dense(128, activation='relu')(dense_d_layer01)
    drop_d_layer0  = BatchNormalization()(dense_d_layer02)
    dense_d_layer1 = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3]*49, activation='relu')(drop_d_layer0)
    drop_d_layer1  = BatchNormalization()(dense_d_layer1)
    rshape_d_layer2 = Reshape((conv_shape[1]*7, conv_shape[2]*7, conv_shape[3]))(drop_d_layer1)
    conv_d_layer3 = Conv2DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2))(rshape_d_layer2)
    drop_d_layer3  = BatchNormalization()(conv_d_layer3)
    conv_d_layer4 = Conv2DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2))(drop_d_layer3)
    conv_d_layer8 = Conv2DTranspose(input_shape[2], 3, padding='same', activation='sigmoid', name='decoder_output')(conv_d_layer4)
    # Define and summarize decoder model
    decoder = Model(decoder_input, conv_d_layer8, name='decoder')
    # print(decoder.summary())
    # return decoder
    # =================
    # VAE as a whole
    # =================
    # Instantiate VAE
    vae_outputs = decoder(encoder(in_e_layer)[2])
    vae         = Model(in_e_layer, vae_outputs, name='vae')

    def kl_reconstruction_loss(true, pred):
        img_width, img_height = xtrain.shape[1], xtrain.shape[2]
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(KB.flatten(true), KB.flatten(pred)) * img_width * img_height
        # KL divergence loss
        kl_loss = 1 + sigma - KB.square(mu) - KB.exp(sigma)
        kl_loss = KB.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return KB.mean(reconstruction_loss + kl_loss)

    vae.compile(optimizer='adam', loss=kl_reconstruction_loss)
    vae.summary()
    return encoder, decoder, vae
