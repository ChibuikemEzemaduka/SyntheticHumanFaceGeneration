import math
import pickle
import random
import numpy as np
from numpy import random
import pathlib
import pickle
import matplotlib.pyplot as plt
import time
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from sklearn.utils import shuffle

#Load data
real_images = np.load('images.npy')
attributes = np.load('attributes5.npy')

#Create generator architecture
def generator_model(z_dim):
    z_random = tf.keras.layers.Input(shape=(z_dim,))
    condition = tf.keras.layers.Input(shape=(5,))
    deconv1 = tf.keras.layers.Dense(6400)(z_random)
    leak1 = tf.keras.layers.LeakyReLU(0.2)(deconv1)
    reshape1 = tf.keras.layers.Reshape((4,4,400))(leak1)
    deconv2 = tf.keras.layers.Embedding(3,65, input_length=5)(condition)
    leak2 = tf.keras.layers.Flatten()(deconv2)
    leak2a = tf.keras.layers.Dense(320)(leak2)
    reshape2 = tf.keras.layers.Reshape((4,4,20))(leak2a)
    concat1 = tf.keras.layers.concatenate([reshape1, reshape2])
    deconv3 = tf.keras.layers.Conv2DTranspose(512, kernel_size=5, strides=2, padding='same')(concat1)
    leak3 = tf.keras.layers.LeakyReLU(0.2)(deconv3)
    deconv4 = tf.keras.layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(leak3)
    leak4 = tf.keras.layers.LeakyReLU(0.2)(deconv4)
    deconv5 = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(leak4)
    leak5 = tf.keras.layers.LeakyReLU(0.2)(deconv5)
    deconv6 = tf.keras.layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(leak5)
    deepgen = tf.keras.Model(inputs=[z_random,condition], outputs=deconv6)
    deepgen.summary()
    return deepgen

#Create discriminator architecture
def discriminator_model():
    real_x = tf.keras.layers.Input(shape=(64,64,3))
    condition1 = tf.keras.layers.Input(shape=(5,))
    deconv2 = tf.keras.layers.Embedding(3,65, input_length=5)(condition1)
    leak2a = tf.keras.layers.Flatten()(deconv2)
    condition2 = tf.keras.layers.Dense(12288)(leak2a)
    condition = tf.keras.layers.Reshape((64,64,3))(condition2)
    concat1 = tf.keras.layers.concatenate([real_x, condition])
    conv3 = tf.keras.layers.Conv2D(256, kernel_size=5, strides=2)(concat1)
    leak3 = tf.keras.layers.LeakyReLU(0.2)(conv3)
    conv4 = tf.keras.layers.Conv2D(512, kernel_size=5, strides=2)(leak3)
    leak4 = tf.keras.layers.LeakyReLU(0.2)(conv4)
    conv5 = tf.keras.layers.Conv2D(1024, kernel_size=3, strides=2)(leak4)
    leak5 = tf.keras.layers.LeakyReLU(0.2)(conv5)
    flatten1 = tf.keras.layers.Flatten()(leak5)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(flatten1)
    deepdisc = tf.keras.Model(inputs=[real_x,condition1], outputs=output)
    deepdisc.summary()
    return deepdisc

#Function to combine generator and discriminator models
def Combined_GAN(deepgen, deepdisc):
     deepdisc.trainable = False
     z_random, condition = deepgen.input
     counterfeit = deepgen.output
     counterfeit_predict = deepdisc([counterfeit, condition])
     deepcombined = tf.keras.Model(inputs=[z_random,condition], outputs=counterfeit_predict)
     deepcombined.summary()
     return deepcombined

#GAN training function
def GAN_trainer(images, attribute, epoch_no, batch_size, rate, z_dim):
    deepgen = generator_model(z_dim)
    deepdisc = discriminator_model()
    deepdisc.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=rate, beta_1=0.5))
    deepcomb = Combined_GAN(deepgen, deepdisc)
    deepcomb.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=rate, beta_1=0.5))
    tf.keras.utils.plot_model(deepgen, to_file='model_plot.png', show_shapes=True, show_layer_names=True,
                              show_layer_activations=True)
    tf.keras.utils.plot_model(deepdisc, to_file='model_plot2.png', show_shapes=True, show_layer_names=True,
                              show_layer_activations=True)
    tf.keras.utils.plot_model(deepcomb, to_file='model_plot3.png', show_shapes=True, show_layer_names=True,
                              show_layer_activations=True)
    discloss, genloss = [], []
    count = 0
    for i in range(epoch_no):
        iter = 0
        images, attribute = shuffle(images, attribute)
        train_dataset = (tf.data.Dataset.from_tensor_slices((images, attribute)).batch(batch_size))
        train_dataset = (train_dataset.map(lambda x, y:
                             (tf.divide(tf.cast(x, tf.float32) - 127.5, 127.5), tf.cast(y, tf.float32))))
        for x,y in train_dataset:
            zed = tf.random.normal([len(x), z_dim, 1], 0, 1)
            for j in range(1):
                x_fake = deepgen.predict([zed, y])
                disc_loss1 = deepdisc.train_on_batch([x,y],tf.ones([len(x),1]))
                disc_loss2 = deepdisc.train_on_batch([x_fake,y],tf.zeros([len(x),1]))
                disc_loss = (disc_loss1 + disc_loss2) / 2
            for j in range(1):
                gen_loss = deepcomb.train_on_batch([zed, y],tf.ones([len(zed),1]))
            print("real data loss is: ", disc_loss1)
            print("fake data loss is: ", disc_loss2)
            print("generator loss is: ", gen_loss)
            print(" ")
            iter += 1
            print ("iter:", iter)
        discloss.append(disc_loss)
        genloss.append(gen_loss)
        count += 1
        print ("epoch: ", count)
    fake_images = deepgen([zed, y])
    deepgen.save("trained_model")
    deepdisc.save("trained_disc")
    deepgen.save('trained_model.h5')
    np.save('disc_loss', discloss)
    np.save('gen_loss', genloss)
    return discloss, genloss, fake_images

epoch_no = 30
batch_size = 128
rate = 0.0001
z_dim = 100
discloss, genloss, fake_images = GAN_trainer(real_images, attributes, epoch_no, batch_size, rate, z_dim)











