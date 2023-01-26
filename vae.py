import tensorflow as tf
from keras.layers import Dense, Conv2D, Input, Conv2DTranspose, Flatten, Reshape, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K
import numpy as np


class VAE(tf.keras.Model):
    def __init__(self, latent_dim= 2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = (28,28,1)
        self.encoder, self.shape_before_flattening = self.create_encoder_CNN_pool()
        self.decoder = self.create_decoder_CNN_pool()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]
    
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encod(data)
            z = VAE.sampler(z_mean, z_log_var)
            reconstruction = self.decod(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, [reconstruction, 1*reconstruction, 1*reconstruction]),
                axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
    
    def create_encoder_CNN_pool(self):
        input_1 = Input(shape=self.image_size, name="input_1")
        x_1 = Conv2D(32, 3, strides = 2, padding = 'same', activation='relu')(input_1)
        x_1 = Conv2D(64, 3, strides = 2, padding = 'same', activation='relu')(x_1)
        
        input_2 = Input(shape=self.image_size, name="input_2") 
        x_2 = Conv2D(32, 3, strides = 2, padding = 'same', activation='relu')(input_2)
        x_2 = Conv2D(64, 3, strides = 2, padding = 'same', activation='relu')(x_2)
        
        input_3 = Input(shape=self.image_size, name="input_3")
        x_3 = Conv2D(32, 3, strides = 2, padding = 'same', activation='relu')(input_3)
        x_3 = Conv2D(64, 3, strides = 2, padding = 'same', activation='relu')(x_3)

        x = concatenate([x_1, x_2, x_3])
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(3, activation="relu", name="bottleneck")(x)

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_model = Model([input_1, input_2, input_3], [z_mean, z_log_var], name="encoder")
        #encoder_model.summary()
        plot_model(encoder_model, "encoder.png", show_shapes=True) 
        return encoder_model, shape_before_flattening
    

    def create_decoder_CNN_pool(self):
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = Dense(np.prod(self.shape_before_flattening[1:]), activation='relu')(decoder_inputs)
        x = Reshape((self.shape_before_flattening[1:]))(x)
        x = Conv2DTranspose(64, 3, strides = 2, padding = 'same', activation='relu')(x)
        x = Conv2DTranspose(32, 3, strides = 2, padding = 'same', activation='relu')(x)
        decoder_outputs = Conv2D(1, 3, padding = 'same', activation='sigmoid')(x)
        decoder_model = Model(decoder_inputs, decoder_outputs, name="decoder")
        #decoder_model.summary()
        plot_model(decoder_model, "decoder.png", show_shapes=True)
        return decoder_model
        
    def decod(self,z):
            return self.decoder(z)
            
    def encod(self,z):
            return self.encoder(z)

    def sampler(z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def sample_image(self,n):
        z_sample = tf.random.normal(shape=(n,self.latent_dim))
        return self.decod(z_sample).numpy()

    
    def save_model(self,name):
        string_enc = name + '_{}d_encoder.h5'.format(self.latent_dim)
        string_dec = name + '_{}d_decoder.h5'.format(self.latent_dim)
        self.encoder.save(string_enc)
        self.decoder.save(string_dec)

    def load_model(self,name):
        string_enc = name + '_{}d_encoder.h5'.format(self.latent_dim)
        string_dec = name + '_{}d_decoder.h5'.format(self.latent_dim)
        self.encoder = tf.keras.models.load_model(string_enc, compile=False)
        self.decoder = tf.keras.models.load_model(string_dec, compile=False)

    def cluster(self, data):
        for i in range(m):
            x_hat = x + eta*tf.random.normal(self.image_size)
            z_mean,z_log_var = self.encod(x_hat)
            z = VAE.sampler(z_mean,z_log_var)
            x_rec = self.decod(z)
            x = x - gamma*(x_hat-x_rec)
        return x

    def clustering(self, inputs, clusters):
        alpha = 1.0
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - clusters), axis=2) / alpha))
        q **= (alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    

