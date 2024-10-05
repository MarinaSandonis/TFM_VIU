import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
from VAE import ConvAutoencoder
import optuna
import random
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics.pairwise import polynomial_kernel
from FID_test import FIDCallback
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
#Hiperparámetros
layers_num = 6
z_dim = 200
dropout_rate = 0.2
lr_rate = 0.0001
num_filters = 16
size_kernel = 3
stride_size = 2
optimizer = "Adamax"
my_CAE = ConvAutoencoder(input_dim, 
                        layers_num,
                        stride_size,
                        z_dim,
                        dropout_rate, 
                        size_kernel,
                        num_filters)
filepath="Best_model_trial_Weights_CheckPoint_3.h5"		
print('Loading ',filepath,' ....')


encoder, decoder, autoencoder = my_CAE.build(use_batch_norm=True, use_dropout=True)
autoencoder.load_weights(filepath)

example_batch = test_generator.__getitem__(index=0)
y_true = example_batch[0]

grid_width, grid_height = (10, 1)
z_sample = np.random.normal(size=(grid_width * grid_height, z_dim))
y_pred = decoder.predict(z_sample)
print(y_pred.shape)

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# # Output the grid of faces
for i in range(grid_width * grid_height):
    z_sample = np.random.normal(size=(1, z_dim))
    y_pred = decoder.predict(z_sample)
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.imshow(y_pred[0, :, :], cmap='gray')
fig.savefig('Best_VAE_Found.png', bbox_inches='tight')
plt.show()


fid = FID_callback.calculate_fid(decoder, test_generator, z_dim, len(df_test), 128)
print('FID value', fid)

ssim_values = []
for i in range(len(df_test)):
    real_batch = test_generator.__getitem__(index=i)
    img_real = np.squeeze((real_batch[0]* 255.0).astype("uint8"))
    z_sample = np.random.normal(size=(1, z_dim))
    img_gen = decoder.predict(z_sample)
    img_gen = np.squeeze((img_gen[0]* 255.0).astype("uint8"))
    print(img_gen.max(), img_gen.min(), img_real.max(), img_real.min())
    ssim_values.append(ssim(img_real, img_gen, data_range=255))
    print('Image ', i, ' SSIM: ',ssim_values[i])

print(np.mean(ssim_values))

