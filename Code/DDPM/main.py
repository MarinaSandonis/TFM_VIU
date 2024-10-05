
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


####################################################
#Hiperparámetros
batch_size = 8
DATASET_REPETITIONS = 1

# NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 30
WEIGHT_DECAY = 1e-4

# optimization
ema = 0.9995
input_dim = (128,128,1)

train_data = utils.image_dataset_from_directory("C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado",
                                        color_mode='grayscale',
                                        labels=None,
                                        image_size=(128, 128),
                                        batch_size=None,
                                        shuffle=True,
                                        seed=42,
                                        interpolation="bilinear")

# Build the U-Net

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

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(num_images=self.num_img,diffusion_steps=PLOT_DIFFUSION_STEPS,).numpy()
        display(generated_images,save_to="C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n4_Diffusion_Buena\\output_ReduceOnPlateau\\generated_img_%03d.png" % (epoch))

def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train = train_data.map(lambda x: preprocess(x))
train = train.repeat(DATASET_REPETITIONS)
train = train.batch(batch_size, drop_remainder=True)

###########################################
def save_trial_info(study, trial):
    with open('trials_info.txt', 'a') as f:
        f.write(f"Trial number: {trial.number}\n")
        f.write(f"Value: {trial.value}\n")
        f.write("Params: \n")
        
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("User attributes:\n")
        for key, value in trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("System attributes:\n")
        for key, value in trial.system_attrs.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"Epoch number: {trial.user_attrs.get('epoch', 'N/A')}\n")
        
        f.write("\n" + "-"*40 + "\n\n")  # Separador para legibilidad

###########################################

####################################################
def objective(trial):

    tf.keras.backend.clear_session()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax", "AdamW"])
    lr_rate = trial.suggest_categorical("lr_prob",  [1e-5, 5e-4, 1e-4, 5e-3, 1e-3])
    scheduler =  trial.suggest_categorical("scheduler", ["Cosine", "Offset", "Linear"]) 

    if scheduler == "Cosine":
        diffusion_schedule = schedulers.cosine_diffusion_schedule
    elif scheduler == "Offset":
        diffusion_schedule = schedulers.offset_cosine_diffusion_schedule
    else:
        diffusion_schedule = schedulers.linear_diffusion_schedule


    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate=lr_rate)
    elif optimizer_name == "Adamax":
        optimizer = Adamax(learning_rate=lr_rate)
    else:
        optimizer = AdamW(learning_rate=lr_rate)


    ddm = DiffusionModel(unet, diffusion_schedule, input_dim, batch_size, ema)

    ddm.normalizer.adapt(train)

    ddm.compile(optimizer=optimizer, loss=losses.mean_absolute_error) #

    model_checkpoint_callback = callbacks.ModelCheckpoint(
                                        filepath="C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n4_Diffusion_Buena\\output_ReduceOnPlateau\\checkpoint.ckpt",
                                        save_weights_only=True,
                                        save_freq="epoch",
                                        verbose=1)

    tensorboard_callback = callbacks.TensorBoard(log_dir="C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n4_Diffusion_Buena\\output_ReduceOnPlateau\\")
    image_generator_callback = ImageGenerator(num_img=10)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='n_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)


    history = ddm.fit(train,
                        epochs=70,
                        callbacks=[model_checkpoint_callback,tensorboard_callback,image_generator_callback, reduce_lr],
                        verbose =1)

    K.clear_session() 
    
    return history.history["val_loss"][-1]
  
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=9, callbacks=[save_trial_info])

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


print('Best hyperparams found by Optuna: \n', study.best_params)
