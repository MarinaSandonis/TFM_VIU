import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Lambda, Dense, \
    Activation, Dropout, Reshape, Input
from tensorflow.keras.models import Model


def Discriminator(input_shape, layers_num, num_filters, size_kernel, stride_size, dropout_rate, use_batch_norm=False,
                  use_dropout=False):
    discriminator_input = Input(shape=input_shape, name='discriminador_input')
    x = discriminator_input
    for i in range(0, layers_num):  # len(encoder_conv_filters)
        conv_layer = Conv2D(filters=num_filters * (2 ** i),
                            kernel_size=(size_kernel, size_kernel),
                            strides=(stride_size, stride_size),
                            padding='same',
                            name='discriminator_conv' + str(i))
        x = conv_layer(x)
        if use_batch_norm & i != 0:
            x = BatchNormalization()(x)

        x = LeakyReLU(alpha=0.2)(x)

        if use_dropout:
            x = Dropout(dropout_rate)(x)
        if i == layers_num - 1:
            x = Conv2D(1, kernel_size=size_kernel, strides=stride_size, padding="same", use_bias=False,
                       activation='sigmoid')(x)
            discriminator_output = Flatten()(x)
    return Model(discriminator_input, discriminator_output)


def Generator(latent_dim, layers_num, num_filters, size_kernel, stride_size, img_shape, use_batch_norm=False):
    generator_input = Input(shape=(latent_dim,))
    output_size = (img_shape[0] // (2 ** layers_num)) * (img_shape[0] // (2 ** layers_num)) * (
                2 * num_filters * (2 ** (layers_num - 1)))
    x = Dense(output_size, use_bias=False)(generator_input)
    x = Reshape((img_shape[0] // (2 ** layers_num),
                 img_shape[0] // (2 ** layers_num),
                 2 * num_filters * (2 ** (layers_num - 1))))(x)
    for i in range(0, layers_num):  # len(encoder_conv_filters)
        conv_layer = Conv2DTranspose(filters=num_filters * (2 ** (layers_num - i - 1)),
                                     kernel_size=(size_kernel, size_kernel),
                                     strides=(stride_size, stride_size),
                                     padding='same',
                                     name='generator_conv' + str(i))
        x = conv_layer(x)
        if use_batch_norm & i != 0:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    generator_output = Conv2DTranspose(img_shape[2], kernel_size=size_kernel, strides=1, activation='tanh',
                                       padding='same')(x)
    return Model(generator_input, generator_output)