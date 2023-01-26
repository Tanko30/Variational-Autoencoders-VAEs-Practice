import tensorflow as tf
import numpy as np
from torch import batch_norm
from vae import VAE
from keras.datasets import mnist
import keras.callbacks as C

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

(x_train, y_train),(x_test, y_test) = mnist.load_data()

training_0 = np.where(np.any([y_train == 0], axis = 0))
training_1 = np.where(np.any([y_train == 1], axis = 0))
training_2 = np.where(np.any([y_train == 2], axis = 0))

testing_0 = np.where(np.any([y_test == 0], axis = 0))
testing_1 = np.where(np.any([y_test == 1], axis = 0))
testing_2 = np.where(np.any([y_test == 2], axis = 0))


x_train_0 = (np.expand_dims(x_train, -1).astype('float32') / 255.)[training_0]
x_train_1 = (np.expand_dims(x_train, -1).astype('float32') / 255.)[training_1]
x_train_2 = (np.expand_dims(x_train, -1).astype('float32') / 255.)[training_2]


x_test_0 = (np.expand_dims(x_test, -1).astype('float32') / 255.)[testing_0]
x_test_1 = (np.expand_dims(x_test, -1).astype('float32') / 255.)[testing_1]
x_test_2 = (np.expand_dims(x_test, -1).astype('float32') / 255.)[testing_2] 


zero_digits = np.concatenate([x_train_0, x_test_0], axis=0)[:6903]
un_digits = np.concatenate([x_train_1, x_test_1], axis=0)[:6903]
two_digits = np.concatenate([x_train_2, x_test_2], axis=0)[:6903]

mc = C.ModelCheckpoint(filepath='output/0_1_2',
                        save_weights_only=True,
                        monitor='total_loss',
                        mode='min',
                        save_best_only=True)
es = C.EarlyStopping(monitor='total_loss', patience=5)
rp = C.ReduceLROnPlateau(monitor='total_loss', patience=5, factor=0.1)

vae = VAE()
vae.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)
vae.fit([zero_digits, un_digits, two_digits], batch_size=32, epochs=1000, callbacks=[mc, es, rp])
#vae.fit(
#    {"input_1": np.array(mnist_digits), "input_2":np.array(mnist_digits), "input_3": np.array(mnist_digits)},
#    epochs=30 ,
#   batch_size=32,
#)

vae.save_model("output/0_1_2_model")