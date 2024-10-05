
import tensorflow as tf
import numpy as np
import random
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class FIDCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, data_generator, latent_dim, num_images=1000, batch_size=16):
        super(FIDCallback, self).__init__()
        self.generator = generator
        self.data_generator = data_generator 
        self.latent_dim = latent_dim
        self.num_images = num_images 
        self.batch_size = batch_size  
        self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def calculate_fid(self, real_images, generated_images):

        num_batches = len(real_images) // self.batch_size
        
        mu_real_total = np.zeros(2048)
        sigma_real_total = np.zeros((2048, 2048))
        mu_gen_total = np.zeros(2048)
        sigma_gen_total = np.zeros((2048, 2048))

        for i in range(num_batches):
            real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size]
            real_batch =  tf.cast((real_batch + 127.5) * 127.5, tf.uint8)
            gen_batch = generated_images[i * self.batch_size:(i + 1) * self.batch_size]
            gen_batch =  tf.cast((gen_batch + 127.5) * 127.5, tf.uint8)

            real_batch_resized = tf.image.resize(real_batch, (299, 299))
            real_batch_resized = tf.image.grayscale_to_rgb(real_batch_resized)
            gen_batch_resized = tf.image.resize(gen_batch, (299, 299))
            gen_batch_resized = tf.image.grayscale_to_rgb(gen_batch_resized)

            real_batch_resized = preprocess_input(real_batch_resized)
            gen_batch_resized = preprocess_input(gen_batch_resized)

            real_features = self.inception_model.predict(real_batch_resized, verbose=0)
            gen_features = self.inception_model.predict(gen_batch_resized, verbose=0)

            if i == 0:
                real_features_tot = real_features
                gen_features_tot = gen_features
            else:
                real_features_tot = np.concatenate((real_features_tot, real_features), axis=0)
                gen_features_tot = np.concatenate((gen_features_tot, gen_features), axis=0)


        mu_real_total = np.mean(real_features_tot, axis=0)
        sigma_real_total = np.cov(real_features_tot, rowvar=False)
        mu_gen_total = np.mean(gen_features_tot, axis=0)
        sigma_gen_total = np.cov(gen_features_tot, rowvar=False)

        ssdiff = np.sum((mu_real_total - mu_gen_total) ** 2.0)
        covmean, _ = sqrtm(sigma_real_total.dot(sigma_gen_total), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real_total + sigma_gen_total - 2.0 * covmean)
        return fid

    def get_real_images(self, num_images):

        real_images = []
        batch_size = self.data_generator.batch_size
        total_batches = len(self.data_generator)  
        batches_needed = int(np.ceil(num_images / batch_size))  

        random_indices = random.sample(range(total_batches), batches_needed)

        for idx in random_indices:
            batch = self.data_generator[idx] 
            real_images.append(batch)

        real_images = np.concatenate(real_images, axis=0)[:num_images]

        return real_images

    def on_epoch_end(self, epoch, logs=None):
        real_images = self.get_real_images(self.num_images)

        latent_vectors = tf.random.normal(shape=(self.num_images, self.latent_dim))
        generated_images = self.generator(latent_vectors, training=False)

        fid = self.calculate_fid(real_images, generated_images)
        print(f'\nFID at epoch {epoch+1}: {fid:.4f}')

        logs['fid'] = fid
        if 'fid' not in self.model.metrics_names:
            self.model.metrics_names.append('fid')

