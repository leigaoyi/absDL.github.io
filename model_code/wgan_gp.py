# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

#from keras.layers.merge import _Merge
#import matplotlib.pyplot as plt
import keras.backend as K
import sys

import numpy as np
import tensorflow as tf
from skimage import io

from config import args

class RandomWeightedAverage():
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def _compute_gradients(tensor, var_list): # solve the problem about tf.gradient, tensorflow 2.0 compatible
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

class WGANGP():
    def __init__(self):
        self.img_rows = args.img_size
        self.img_cols = args.img_size
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = args.img_size

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,self.latent_dim, 1))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc) # predict images

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        alpha = K.random_uniform((args.batch_size, args.img_size, args.img_size, 1))
        #interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        interpolated_img = alpha * real_img + (1 - alpha) * fake_img
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,self.latent_dim, 1))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        #gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients = _compute_gradients(y_pred, [averaged_samples])[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    # def build_generator(self):
    #
    #     model = Sequential()
    #
    #     model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
    #     model.add(Reshape((7, 7, 128)))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(128, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(64, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
    #     model.add(Activation("tanh"))
    #
    #     model.summary()
    #
    #     noise = Input(shape=(self.latent_dim,))
    #     img = model(noise)
    #
    #     return Model(noise, img)

    def build_generator(self, input_size=(args.img_size, args.img_size, 1)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation=None)(conv9)
        #conv10 = Reshape((256, 256))(conv10)
        model = Model(inputs, conv10)

        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        # # model.summary()
        # if (pretrained_weights):
        #     model.load_weights(pretrained_weights)
        return model

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # def train(self, epochs, batch_size, sample_interval=50):
    #
    #     # Load the dataset
    #     (X_train, _), (_, _) = mnist.load_data()
    #
    #     # Rescale -1 to 1
    #     X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #     X_train = np.expand_dims(X_train, axis=3)
    #
    #     # Adversarial ground truths
    #     valid = -np.ones((batch_size, 1))
    #     fake =  np.ones((batch_size, 1))
    #     dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
    #     for epoch in range(epochs):
    #
    #         for _ in range(self.n_critic):
    #
    #             # ---------------------
    #             #  Train Discriminator
    #             # ---------------------
    #
    #             # Select a random batch of images
    #             idx = np.random.randint(0, X_train.shape[0], batch_size)
    #             imgs = X_train[idx]
    #             # Sample generator input
    #             noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
    #             # Train the critic
    #             d_loss = self.critic_model.train_on_batch([imgs, noise],
    #                                                             [valid, fake, dummy])
    #
    #         # ---------------------
    #         #  Train Generator
    #         # ---------------------
    #
    #         g_loss = self.generator_model.train_on_batch(noise, valid)
    #
    #         # Plot the progress
    #         print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
    #
    #         # If at save interval => save generated image samples
    #         # if epoch % sample_interval == 0:
    #         #     self.sample_images(epoch)

    def train_on_batch(self, x, y=None,batch_size=args.batch_size):
        '''
        x: noise
        y: unmasked
        '''
        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        
        x = np.reshape(x, (args.batch_size, args.img_size, args.img_size, 1))
        y = np.reshape(y, (args.batch_size,args.img_size,args.img_size,1))
        #print('x shape ', x.shape)
        for _ in range(self.n_critic):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            #idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = y
            # Sample generator input
            noise = x
            # Train the critic
            d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                            [valid, fake, dummy])
        # ---------------------
        #  Train Generator
        # ---------------------
        g_loss = self.generator_model.train_on_batch(noise, valid)

        # Plot the progress
        #print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        # If at save interval => save generated image samples
        # if epoch % sample_interval == 0:
        #     self.sample_images(epoch)
        return g_loss

    def test_on_batch(self, x, y=None,batch_size=args.batch_size):
        '''
        x: noise
        y: unmasked
        '''
        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        for _ in range(self.n_critic):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            #idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = y
            # Sample generator input
            noise = x
            # Train the critic
            d_loss = self.critic_model.test_on_batch([imgs, noise],
                                                            [valid, fake, dummy])
        # ---------------------
        #  Train Generator
        # ---------------------
        g_loss = self.generator_model.test_on_batch(noise, valid)

        # Plot the progress
        #print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        # If at save interval => save generated image samples
        # if epoch % sample_interval == 0:
        #     self.sample_images(epoch)
        return g_loss

    def save_generator(self, path):
        self.generator.save(path, include_optimizer=False)

    def sample_images(self, x, epoch=100):
        #r, c = 5, 5
        noise = x
        gen_imgs = self.generator.predict(noise)

        sample_img = np.reshape(gen_imgs[0], (args.img_size,args.img_size,1))
        max_i = sample_img.max()
        min_i = sample_img.min()
        sample_img = (sample_img - min_i)/(max_i - min_i)

        io.imsave('./tmp/sample_E{}.png'.format(epoch), sample_img)

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/mnist_%d.png" % epoch)
        # plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)