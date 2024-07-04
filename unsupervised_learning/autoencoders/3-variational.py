#!/usr/bin/env python3

"""This module creates a variational autoencoder."""

# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    
    args (tensor): mean and log of variance of Q(z|X)
        
    Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def vae_loss(inputs, outputs, z_mean, z_log_var, input_dims):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a variational autoencoder.

    input_dims (ints): The dimension of the model input.
    hidden_layers (list): List containing the number of nodes for each
        hidden layer in the encoder. reversed for the decoder.
    latent_dims (int): The dimension of the latent space representation.

    Returns:
        tuple: encoder, decoder, auto.
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    all layers use relu activation except for the decoder output which uses
    sigmoid.
    """
    # Encoder
    inputs = layers.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    z_mean = layers.Dense(latent_dims)(x)
    z_log_var = layers.Dense(latent_dims)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs, [z, z_mean, z_log_var], name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    # VAE Model
    outputs = decoder(encoder(inputs)[0])
    auto = Model(inputs, outputs, name='vae')

    # Compile the model with the custom loss
    auto.add_loss(vae_loss(inputs, outputs, encoder(inputs)[1], encoder(inputs)[2], input_dims))
    auto.compile(optimizer='adam')
    
    return encoder, decoder, auto
