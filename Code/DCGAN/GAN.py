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
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        
    # def call(self, inputs): ##DESCOMENTAR EN TEST
    #     generated_images = self.generator(inputs)  
    #     return self.discriminator(generated_images)
        
    def compile(self, optimizer_name_driscri, optimizer_name_gene, learning_rate_d, learning_rate_g): 
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g

        if optimizer_name_driscri == "Adam":
            self.d_optimizer = Adam(learning_rate=self.learning_rate_d)
        elif optimizer_name_driscri == "Adamax":
            self.d_optimizer = Adamax(learning_rate=self.learning_rate_d)
        elif optimizer_name_driscri == "AdamW":
            self.d_optimizer = AdamW(learning_rate=self.learning_rate_d, weight_decay=0.004)

        if optimizer_name_gene == "Adam":
            self.g_optimizer = Adam(learning_rate=self.learning_rate_g)
        elif optimizer_name_gene == "Adamax":
            self.g_optimizer = Adamax(learning_rate=self.learning_rate_g)
        elif optimizer_name_gene == "AdamW":
            self.g_optimizer = AdamW(learning_rate=self.learning_rate_g, weight_decay=0.004)

        super(DCGAN, self).compile()

        self.loss_fn = BinaryCrossentropy()
        self.d_loss_metric = Mean(name="d_loss")
        self.d_real_acc_metric = BinaryAccuracy(name="d_real_acc")
        self.d_fake_acc_metric = BinaryAccuracy(name="d_fake_acc")
        self.d_acc_metric = BinaryAccuracy(name="d_acc")
        self.g_loss_metric = Mean(name="g_loss")
        self.g_acc_metric = BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_real_acc_metric,
            self.d_fake_acc_metric,
            self.d_acc_metric,
            self.g_loss_metric,
            self.g_acc_metric,
        ]
            
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(generated_images, training=True)

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(tf.shape(real_predictions))
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(tf.shape(fake_predictions))

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, fake_predictions)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.d_real_acc_metric.update_state(real_labels, real_predictions)
        self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
        self.d_acc_metric.update_state([real_labels, fake_labels], [real_predictions, fake_predictions] )
        self.g_loss_metric.update_state(g_loss)
        self.g_acc_metric.update_state(real_labels, fake_predictions)

        return {m.name: m.result() for m in self.metrics}
    