import tensorflow as tf
import numpy as np
import random
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class FIDCallback(tf.keras.callbacks.Callback):
    def __init__(self,  decoder, data_generator, z_dim, num_images=1000, batch_size=16):
        super(FIDCallback, self).__init__()
        self.decoder = decoder  # El decoder de la VAE (para generar imágenes)
        self.data_generator = data_generator  # Generador de datos para obtener imágenes reales
        self.num_images = num_images  # Número de imágenes para calcular FID
        self.batch_size = batch_size  # Tamaño del lote para el cálculo del FID
        self.z_dim  =z_dim
        self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def calculate_fid(self, real_images, generated_images):
        """
        Calcula el FID usando imágenes reales y generadas, dividiéndolas en lotes para reducir el uso de memoria.
        """
        num_batches = len(real_images) // self.batch_size
        
        # Inicializar acumuladores de medias y covarianzas
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

            mu_real = np.mean(real_features, axis=0)
            sigma_real = np.cov(real_features, rowvar=False)
            mu_gen = np.mean(gen_features, axis=0)
            sigma_gen = np.cov(gen_features, rowvar=False)

            mu_real_total += mu_real
            sigma_real_total += sigma_real
            mu_gen_total += mu_gen
            sigma_gen_total += sigma_gen

        mu_real_total /= num_batches
        mu_gen_total /= num_batches
        sigma_real_total /= num_batches
        sigma_gen_total /= num_batches

        ssdiff = np.sum((mu_real_total - mu_gen_total) ** 2.0)
        covmean, _ = sqrtm(sigma_real_total.dot(sigma_gen_total), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real_total + sigma_gen_total - 2.0 * covmean)
        return fid

    def get_real_images(self, num_images):
        """
        Obtiene un número especificado de imágenes reales usando el generador de datos.
        """
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
