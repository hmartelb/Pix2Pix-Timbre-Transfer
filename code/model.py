from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import time

import matplotlib.pyplot as plt

import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, 
                                    size, 
                                    strides=2, 
                                    padding='same',
                                    kernel_initializer=initializer, 
                                    use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, 
                                                size, 
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator(input_shape=[None,None,1]):
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), 
        downsample(128, 4), 
        downsample(256, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4), 
        upsample(256, 4), 
        upsample(128, 4), 
        upsample(64, 4), 
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(filters=1, 
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')
    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator(input_shape=[None,None,1]):
    input_img = tf.keras.layers.Input(shape=input_shape)
    generated_img = tf.keras.layers.Input(shape=input_shape)

    con = tf.keras.layers.Concatenate()([input_img, generated_img])
    initializer = tf.random_normal_initializer(0., 0.02)

    down1 = downsample(64, 4, apply_batchnorm=False)(con)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    down4 = downsample(512, 4)(down3)

    last = tf.keras.layers.Conv2D(filters=1, 
                                    kernel_size=4, 
                                    strides=1, 
                                    kernel_initializer=initializer, 
                                    padding='same')(down4)

    return tf.keras.Model(inputs=[input_img, generated_img], outputs=last)

def main():
    generator = Generator()
    generator.summary()

    discriminator = Discriminator()
    discriminator.summary()

if __name__ == '__main__':
    main()
