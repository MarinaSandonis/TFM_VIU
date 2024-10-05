
import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_fid(generator, data_generator, latent_dim, num_images=1000, batch_size=16):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    num_batches = num_images // batch_size
    
    mu_real_total = np.zeros(2048)
    sigma_real_total = np.zeros((2048, 2048))
    mu_gen_total = np.zeros(2048)
    sigma_gen_total = np.zeros((2048, 2048))

    for i in range(num_batches):
        print("Batch " + str(i) + "/"+ str(num_batches))
        real_batch = get_real_images(data_generator, batch_size, i * batch_size, (i + 1) * batch_size)
        real_batch =  tf.cast((real_batch + 127.5) * 127.5, tf.uint8)
        latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        gen_batch = generator(latent_vectors, training=False)
        gen_batch =  tf.cast((gen_batch + 127.5) * 127.5, tf.uint8)

        real_batch_resized = tf.image.resize(real_batch, (299, 299))
        real_batch_resized = tf.image.grayscale_to_rgb(real_batch_resized)
        gen_batch_resized = tf.image.resize(gen_batch, (299, 299))
        gen_batch_resized = tf.image.grayscale_to_rgb(gen_batch_resized)

        real_batch_resized = preprocess_input(real_batch_resized)
        gen_batch_resized = preprocess_input(gen_batch_resized)

        real_features = inception_model.predict(real_batch_resized, steps=1, verbose=1)
        gen_features = inception_model.predict(gen_batch_resized, steps=1, verbose=1)

        if i == 0:
            real_features_tot = real_features
            gen_features_tot = gen_features
        else:
            real_features_tot = np.concatenate((real_features_tot, real_features), axis=0)
            gen_features_tot = np.concatenate((gen_features_tot, gen_features), axis=0)

    np.save('Real_features_tot_concat.npy', real_features_tot)
    np.save('Gen_features_tot_concat.npy', gen_features_tot)

    mu_real_total = np.mean(real_features_tot, axis=0)
    sigma_real_total = np.cov(real_features_tot, rowvar=False)
    mu_gen_total = np.mean(gen_features_tot, axis=0)
    sigma_gen_total = np.cov(gen_features_tot, rowvar=False)

    ssdiff = np.sum((mu_real_total - mu_gen_total) ** 2.0)
    covmean, _ = sqrtm(sigma_real_total.dot(sigma_gen_total), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real_total + sigma_gen_total - 2.0 * covmean)
    real_means = np.mean(real_features_tot, axis=1)
    gen_means = np.mean(gen_features_tot, axis=1)

    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(real_means, label='Reales', fill=True, color="gray", alpha=0.5)
    sns.kdeplot(gen_means, label='Generadas', fill=True, color="orange", alpha=0.5)

    plt.title('Distribución de Activaciones (Promedio)')
    plt.xlabel('Media de Activaciones')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()
    return fid

def get_real_images(data_generator, batch_size, inicio, fin):

    real_images = []
    for idx in range (inicio,  fin):
        batch = data_generator[idx] 
        real_images.append(batch)

    real_images = np.concatenate(real_images, axis=0)[:batch_size]

    return real_images


    
