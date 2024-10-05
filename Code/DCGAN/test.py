import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
import random
import numpy as np
import cv2
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics.pairwise import polynomial_kernel
from GAN import DCGAN
from tensorflow.python.client import device_lib
import modelos_DCGAN
from FID import FIDCallback
import matplotlib.pyplot as plt
import os
import FID_test
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

test_path = 'C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\test_list.txt'
df_test = pd.read_csv(test_path, header=None, names=['ImageID'])

input_dim = (128,128,1)
batch_size=8

# Carga de datos
test_generator = DataGenerator(df_test, 
                                  input_dim[1], 
                                  input_dim[0], 
                                  input_dim[2], 
                                  batch_size=1, 
                                  path_to_img='C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\Images',
                                  shuffle = False) 
tf.keras.backend.clear_session()
optimizer_name_driscri = 'Adamax'
optimizer_name_gene = 'Adam'
layers_num = 4
z_dim = 100
dropout_rate = 0.1
learning_rate_d = 0.001
learning_rate_g = 0.0002
num_filters = 8
stride_size = 2
size_kernel = 4

discriminator = modelos_DCGAN.Discriminator(input_dim,layers_num, num_filters, size_kernel, stride_size, dropout_rate, use_batch_norm = True, use_dropout=True)
discriminator.summary()
generator = modelos_DCGAN.Generator(z_dim, layers_num, num_filters, size_kernel,stride_size, input_dim, use_batch_norm=True) 
generator.summary()
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=z_dim)
dcgan.compile(optimizer_name_driscri, optimizer_name_gene, learning_rate_d, learning_rate_g)


input_shape = (1, z_dim)  
dummy_input = np.random.normal(size=input_shape).astype(np.float32)
dcgan(dummy_input) 


filepath = "Best_DCGAN_Adamaxdiscri_Adamagenera_001lrd_0002lrg_01dropout_100zdim_4layers_8filters_Completo.h5"
dcgan.load_weights(filepath)
print('El número de imagenes de test es: ', len(df_test))

example_batch = test_generator.__getitem__(index=0) 
y_true = example_batch[0]
print(y_true.shape, np.max(y_true), np.min(y_true))

latent_vector = np.random.normal(size=(30, z_dim))
y_pred = generator.predict(latent_vector)
print(y_pred.shape)
y_pred = ((y_pred+ 127.5)*127.5).astype("uint8")
print(y_pred.shape, np.max(y_pred), np.min(y_pred))

for i in range(0, y_pred.shape[0]):
    original = np.squeeze(((y_true[i] +127.5)* 127.5).astype("uint8"))
    recon = np.squeeze(((y_pred[i]+127.5) * 127.5).astype("uint8"))
    cv2.imwrite("Imagen_"+str(i)+".png", recon) 
fid = FID_test.calculate_fid(generator, test_generator, z_dim, len(df_test), 128)
print('FID value', fid)


ssim_values = []
for i in range(len(df_test)):
    real_batch = test_generator.__getitem__(index=i)
    img_real = np.squeeze(real_batch[0])
    latent_vector = np.random.normal(size=(1, z_dim))
    img_gen = generator.predict(latent_vector)
    img_gen = np.squeeze(((img_gen[0]+127.5) * 127.5).astype("uint8"))
    values_range = img_real.max() - img_real.min()
    ssim_values.append(ssim(img_real, img_gen))
    print('Image ', i, ' SSIM: ',ssim_values[i])

print(np.mean(ssim_values))
