from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Lambda, Dense, Activation, Dropout, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.metrics import Mean, BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


class DCGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, critic_steps, gp_weight):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    # def call(self, inputs): #Descomentar en test
    #     generated_images = self.generator(inputs) 
    #     return self.discriminator(generated_images)

        
    def compile(self, d_optimizer, g_optimizer):

        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss_metric = Mean(name="d_wass_loss")
        self.d_gp_metric = Mean(name="d_gp")
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_wass_loss_metric,
            self.d_gp_metric,
            self.g_loss_metric,
        ]
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        fake_images = tf.cast(fake_images, dtype=tf.float32)
        real_images = tf.cast(real_images, dtype=tf.float32)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_predictions = self.discriminator(fake_images, training=True)
                real_predictions = self.discriminator(real_images, training=True)

                d_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                d_gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_wass_loss + d_gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.discriminator(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_wass_loss)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}