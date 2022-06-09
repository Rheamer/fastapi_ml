from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.activation import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


import numpy as np
import matplotlib as plt


class SmileyGAN:

    def build_generator(self):
        generator = Sequential()
        generator.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
        generator.add(Reshape((7, 7, 128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=3, padding="same",
                             activation="relu"))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D())
        # convolutional + batch normalization layers
        generator.add(Conv2D(64, kernel_size=3, padding="same",
                             activation="relu"))
        generator.add(BatchNormalization(momentum=0.8))
        # convolutional layer with filters = 1
        generator.add(Conv2D(1, kernel_size=3, padding="same",
                             activation="relu"))
        generator.summary()
        noise = Input(shape=(100,))
        fake_image = generator(noise)
        return Model(inputs=noise, outputs=fake_image)

    def build_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        img = Input(shape=(28, 28, 1))
        probability = discriminator(img)
        return Model(inputs=img, outputs=probability)

    def visualize_input(self, img, ax):
        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y] < thresh else 'black')

    def init(self):
        #load numpy smileys
        # training_data =
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)
        visualize_input(training_data[3343], ax)
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        discriminator = build_discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        discriminator.trainable = False
        generator = build_generator()
        z = Input(shape=(100,))
        img = generator(z)
        valid = discriminator(img)
        combined = Model(inputs=z, outputs=valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
