import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import voxelmorph as vxm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import get_model
from data_generator import train_generator, test_generator
import matplotlib.pyplot as plt
import keras.backend as K

def focal_loss(y_true, y_pred):
    #flatten label and prediction tensors
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    
    BCE = K.binary_crossentropy(y_true_pos, y_pred_pos)
    BCE_EXP = K.exp(-BCE)
    
    alpha = 0.7
    gamma = 2
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)
    return focal_loss

# =============================================================================
# Build the backbone model
# =============================================================================

moving_image_shape = (96, 96, 96, 1)
fixed_image_shape = (96, 96, 96, 1)

model = get_model(moving_image_shape, fixed_image_shape, with_label_inputs=False)

print('\nBackbone model inputs and outputs:')

print('    input shape: ', ', '.join([str(t.shape) for t in model.inputs]))
print('    output shape:', ', '.join([str(t.shape) for t in model.outputs]))

# =============================================================================
# Build the registration network
# =============================================================================

# build transformer layer
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

# extract the moving image
moving_image = model.input[0]

# extract ddf
ddf = model.output

# warp the moving image with the transformer using network-predicted ddf
moved_image = spatial_transformer([moving_image, ddf])

outputs = [moved_image, ddf]

registration_model = keras.Model(inputs=model.inputs, outputs=outputs)

print('\nRegistration network inputs and outputs:')

print('    input shape: ', ', '.join([str(t.shape) for t in registration_model.inputs]))
print('    output shape:', ', '.join([str(t.shape) for t in registration_model.outputs]))

losses = [focal_loss, vxm.losses.Grad('l2').loss]
lambda_param = 0.05
loss_weights = [1, lambda_param]

registration_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# =============================================================================
# Training loop
# =============================================================================

f_path = r'/workspace/reg_challenge/dataset/train'

val_path = r'/workspace/reg_challenge/dataset/val'

model_save_path = r'/workspace/reg_challenge/Baseline-Networks/voxelmorph_model_checkpoints_focal_size96'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

batch_size = 4

train_gen = train_generator(f_path, batch_size, moving_image_shape, fixed_image_shape, with_label_inputs=False)

num_trials = 40

val_dice = []

# registration_model = keras.models.load_model(os.path.join(model_save_path, 'registration_model_trial_328'), custom_objects={'loss': [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss], 'loss_weights': [1, lambda_param]})

for trial in range(0, num_trials):
    print(f'\nTrial {trial} / {num_trials-1}:')
    
    hist = registration_model.fit(train_gen, epochs=1, steps_per_epoch=32, verbose=1);
    
    dice_scores = []
    for label_num in range(1):
        val_gen = test_generator(f_path, 4, moving_image_shape, fixed_image_shape, start_index=None, end_index=None, label_num=label_num, with_label_inputs=True)
        while True:
            try:
                (val_inputs, val_outputs) = next(val_gen)
                moving_images_val, fixed_images_val, moving_labels_val, fixed_labels_val = val_inputs
                fixed_images_val, fixed_labels_val, zero_phis_val = val_outputs
                _, ddf_val = registration_model.predict((moving_images_val, fixed_images_val), verbose=0)
                
                moved_labels_val = spatial_transformer([moving_labels_val, ddf_val])
                moved_images_val = spatial_transformer([moving_images_val, ddf_val])
                
                dice_score = np.array(-1.0 * vxm.losses.Dice().loss(tf.convert_to_tensor(moved_labels_val, dtype='float32'), tf.convert_to_tensor(fixed_labels_val, dtype='float32')))
                dice_scores.append(dice_score)
            except (IndexError, StopIteration) as e:
                break
    val_dice.append(np.mean(dice_scores))
    plt.plot(val_dice, 'r')
    plt.xlabel('Trials')
    plt.ylabel('Dice')
    plt.savefig(r'voxelmorph_focalloss_size96_val_dice_2.png')
    print('    Validation Dice: ', np.mean(dice_scores))
    if trial % 4 == 0:
        registration_model.save(os.path.join(model_save_path, f'registration_model_trial_{trial}'))
        print('Model saved!')
