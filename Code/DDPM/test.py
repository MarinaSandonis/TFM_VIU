
import numpy as np
import matplotlib.pyplot as plt
from Diffusion import DiffusionModel
plt.style.use("seaborn-v0_8-colorblind")
import pandas as pd
from tensorflow.keras import backend as K
import math
from my_data_generator import DataGenerator
import tensorflow as tf
from tensorflow.keras import layers,models,callbacks,losses,utils
import random
from utils import DownBlock, UpBlock, ResidualBlock, sinusoidal_embedding, display
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow_addons.optimizers import AdamW
import optuna
import schedulers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import FID_test
from skimage.metrics import structural_similarity as ssim



#Hiperparámetros
batch_size = 8
DATASET_REPETITIONS = 1

PLOT_DIFFUSION_STEPS = 20

# optimization
ema = 0.999
input_dim = (128,128,1)

noisy_images = layers.Input(shape=input_dim)
x = layers.Conv2D(32, kernel_size=1)(noisy_images)

noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=input_dim[0], interpolation="nearest")(noise_embedding)

x = layers.Concatenate()([x, noise_embedding])

skips = []

x = DownBlock(32, block_depth=2)([x, skips])
x = DownBlock(64, block_depth=2)([x, skips])
x = DownBlock(96, block_depth=2)([x, skips])

x = ResidualBlock(128)(x)
x = ResidualBlock(128)(x)

x = UpBlock(96, block_depth=2)([x, skips])
x = UpBlock(64, block_depth=2)([x, skips])
x = UpBlock(32, block_depth=2)([x, skips])

x = layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)

unet = models.Model([noisy_images, noise_variances], x, name="unet")

def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train_data = utils.image_dataset_from_directory(
    "C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\Train",
    color_mode='grayscale',
    labels=None,
    image_size=(128, 128),
    batch_size=None,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

train = train_data.map(lambda x: preprocess(x))
train = train.repeat(DATASET_REPETITIONS)
train = train.batch(batch_size, drop_remainder=True)

diffusion_schedule = schedulers.offset_cosine_diffusion_schedule

ddm = DiffusionModel(unet, diffusion_schedule, input_dim, batch_size, ema)
ddm.normalizer.adapt(train)

ddm.built = True
ddm.load_weights("C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n4_Diffusion_Buena\\output_ReduceOnPlateau\\checkpoint.ckpt")

test_path = 'C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\test_list.txt'
df_test = pd.read_csv(test_path, header=None, names=['ImageID'])
test_generator = DataGenerator(df_test,
                                input_dim[1], 
                                input_dim[0], 
                                input_dim[2], 
                                batch_size=1, 
                                path_to_img="C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\resie_512x512",
                                shuffle = True) 

FID = FID_test.calculate_fid(ddm, test_generator, num_images=len(df_test), batch_size=64)
print(FID)

ssim_values = []
for i in range(100):
    real_batch = test_generator.__getitem__(index=i)
    img_real = np.squeeze(real_batch[0]).astype("uint8")
    img_gen = generated_images = ddm.generate(num_images=1,diffusion_steps=50,).numpy()
    img_gen = np.squeeze((img_gen[0] * 255.0).astype("uint8"))
    values_range = img_real.max() - img_real.min()
    ssim_values.append(ssim(img_real, img_gen))
    print('Image ', i, ' SSIM: ',ssim_values[i])

np.save("SSIM_values.npy", np.array(ssim_values))
print(np.mean(ssim_values))



