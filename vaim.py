import tensorflow as tf
import numpy as np
import os
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, ActivityRegularization, Lambda, Add, concatenate
from tensorflow.keras import regularizers
from keras.layers.merge import _Merge
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import keras
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import initializers
from keras import objectives
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors as mcol
from keras.callbacks import History, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2, l1
from tensorflow.keras import regularizers
import pylab as py
import pandas as pd
from matplotlib.lines import Line2D

class VAIM():
    
    def __init__(self):
    
        # default is set to x2 toy example
        self.example = 'x2'
        self.input_shape = (1,) 
        self.output_shape = 1
        self.latent_dim = 100
        self.encoder_dim = 1024
        self.batch_size = 512
        self.epochs = 2000
        self.l2_reg  = 1e-5
        self.DIR = 'saved_model/'
        checkdir(self.DIR)
        self.history = History()
        
        opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  
        
        inputs = Input(shape = self.input_shape, name='encoder_input')
        
        self.encoder = self.encoder(inputs)
        self.decoder = self.decoder()
        outputs = self.decoder(self.encoder(inputs)[2:4])
   
        self.model = Model(inputs= inputs, outputs= outputs)
        self.model.compile(loss=[self.vae_loss, 'mse'], optimizer=opt, metrics=['mse'])
        
        
    # -- build encoder model
    def encoder(self, inputs):
        x1 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(inputs)
        x2 = LeakyReLU()(x1)
        x2 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x2)
        x2 = LeakyReLU()(x2)
        x3 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x2)
        sc1 = Add()([x1,x3])
        x3 = LeakyReLU()(sc1)
        x4 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x3)
        sc2 = Add()([x2,x4])
        x4 = LeakyReLU()(sc2)
        x5 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x4)
        sc3 = Add()([x3,x5])
        x5 = LeakyReLU()(sc3)
        x6 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x5)
        sc4 = Add()([x4,x6])
        x6 = LeakyReLU()(sc4)
        x7 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x6)
        x7 = LeakyReLU()(x7)
        yy = Dense(self.output_shape, name = 'obs')(x7)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(x7)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x7)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        encoder = Model(inputs= inputs, outputs=[self.z_mean, self.z_log_var, self.z,  yy], name='encoder')
        encoder.summary()

        return encoder  
        
    # -- decoder model
    def decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        yyy = Input(shape=self.input_shape, name='yyy')
        con = concatenate([latent_inputs, yyy], axis = 1)
        x1 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(con)
        x1 = LeakyReLU()(x1)
        x2 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x1)
        x2 = LeakyReLU()(x2)
        x3 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x1)
        sc1 = Add()([x1,x3])
        x3 = LeakyReLU()(sc1)
        x4 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x3)
        sc2 = Add()([x2,x4])
        x4 = LeakyReLU()(sc2)
        x5 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x4)
        sc3 = Add()([x3,x5])
        x5 = LeakyReLU()(sc3)
        x6 = Dense(self.encoder_dim, kernel_regularizer=l2(self.l2_reg))(x5)
        sc4 = Add()([x4,x6])
        x6 = LeakyReLU()(sc4)
        out = Dense((self.output_shape))(x6)

        decoder = Model(inputs=[latent_inputs, yyy], outputs= [out, yyy] , name='decoder')
        decoder.summary()

        return decoder    
        
    # -- sampling function
    def sampling(self, args):
        
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        self.z_mean, self.z_log_var = args
        batch = K.shape(self.z_mean)[0]
        dim = K.int_shape(self.z_mean)[1]
        epsilon = K.random_uniform(shape=(batch, dim))
            
        return self.z_mean + K.exp(0.5 * self.z_log_var) * epsilon
    
    # loss function
    def vae_loss(self, inputs, outputs):
    
        mse_loss = objectives.mse(inputs, outputs)
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        loss = K.mean(kl_loss + mse_loss)
        return loss
        
    # -- Train the model
    def train(self, X_train, y_train):
        
        if self.example == 'x2':
            self.model_name = 'x2model.hd5'
        elif self.example == 'sin':
            self.model_name = 'sinmodel.hd5'
            
        checkpointer = ModelCheckpoint(filepath = self.DIR + self.model_name, verbose=1, save_best_only=True)
        self.model.fit(x = [X_train], y = [X_train, y_train], epochs= self.epochs, batch_size= self.batch_size,validation_split=0.3,callbacks = [checkpointer, self.history])
        return self.history
    
    # -- predict using test sets
    def predict(self, vae, X_train, y_test):
        latent_mean, latent_logvar, Z  = vae.encoder.predict(X_train)[0:3]
        latent_var = np.exp(latent_logvar)
        latent_std = np.sqrt(latent_var)

        # -- sample using latent mean and std
        SAMPLE_SIZE = y_test.shape[0]
        z_samples = np.empty([SAMPLE_SIZE, self.latent_dim])

        for i in range(0,SAMPLE_SIZE):
            for j in range(0, self.latent_dim):
                z_samples[i,j] = np.random.uniform(latent_mean[i%SAMPLE_SIZE, j], latent_std[i%SAMPLE_SIZE,j])

        # -- predict using the samples generated by the encoder and y_test
        results = vae.decoder.predict([z_samples, y_test])
        return results

    # -- get latent z
    def get_latent(self,vae, X_train):
        Z  = vae.encoder.predict(X_train)[2]
        return Z

 # -- check if direcorty exists, otherwise create one
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

# -- Generate toy data for sin function
def generate_sin_samples(N = 1000, domain = 4):

    x = (np.random.rand(N, 1)-0.5) * domain * np.pi
    y = np.sin(x) + np.random.randn(N, 1) * 0.05

    return x, y

# -- Generate toy data for f(x) = x^2
def generate_x2_samples(N = 1000, noise = 0.05, domain = 5):

    x = (np.random.rand(N, 1) -0.5 ) * (domain * 2)
    y = np.power(x, 2) + np.random.randn(N, 1) * noise

    return x, y

# -- Generate toy data for 2d circle example f(x) = x1^2+x2^2
def generate_x2_y2(N = 1000):

    x = np.random.randn(N, 2)
    y = x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1] 

    return x, y

# -- plot function of sin
def plot_sin(x,y):
    plt.plot(x, y, '.')
    plt.xlabel('x', size = 12)
    plt.ylabel(r'$\sin(x)$', size = 12)
    plt.show()

# -- plot function of f(x) = x^2
def plot_x2(x,y):
    plt.plot(x, y, '.')
    plt.xlabel('x', size = 12)
    plt.ylabel(r'$x^2$', size = 12)
    plt.show()

# -- plot loss
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.semilogy()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# -- plot results
def plot_result(result, X_test, y_test):
    fig, ax = plt.subplots()
    ax.plot(X_test, y_test, '.')
    ax.plot(result[0] , y_test, '.')
    ax.legend(['true', 'pred'])
    plt.savefig('result.png')

# -- plot latent
def plot_latent(Z, X_train):
    pca = PCA(n_components=2)
    x = pca.fit_transform(Z)
    fig, ax = plt.subplots()
    im = ax.scatter(x[:,0], x[:,1],cmap='jet',c = X_train.reshape(-1), s = 2)
    cb = fig.colorbar(im)
    cb.set_label(r'$x$', labelpad=-26, y=1.07, rotation=0, size = 12)
    ax.set_xlabel('$PCA\ 1$')
    ax.set_ylabel('$PCA\ 2$')
    plt.savefig('latent.png')
