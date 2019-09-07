from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from config import (CHECKPOINT_DIR, DATASET_PATH, IMG_DIM, OUTPUT_PATH,
                    TEST_AUDIOS_PATH)
from data import (DataGenerator, amplitude_to_db, db_to_amplitude,
                  forward_transform, init_directory, inverse_transform,
                  join_magnitude_slices, load_audio, slice_magnitude,
                  write_audio)
from losses import discriminator_loss, generator_loss
from model import Discriminator, Generator


def generate_audio(prediction, phase, output_name):
    mag_db = join_magnitude_slices(prediction, phase.shape)
    mag = db_to_amplitude(mag_db)
    audio = inverse_transform(mag, phase)
    write_audio(output_name, audio)

def generate_images(prediction, test_input, target, output_name):
    display_list = [test_input[0,:,:,0], target[0,:,:,0], prediction[0,:,:,0]]
    title = ['input', 'true', 'pred']
    for i in range(3):
        temp_img = np.flip((display_list[i] + 1) / 2, axis=0) # [-1,1] >> [0,1]
        plt.imsave(output_name+'_'+title[i]+'.png', temp_img) 

def write_csv(df, output_name):
    df.to_csv(output_name, header='column_names')

def plot_loss_findlr(losses, lrs, output_name, n_skip_beginning=10, n_skip_end=5):
    """
    Plots the loss.
    Parameters:
        n_skip_beginning - number of batches to skip on the left.
        n_skip_end - number of batches to skip on the right.
    """
    plt.figure()
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(lrs[n_skip_beginning:-n_skip_end], losses[n_skip_beginning:-n_skip_end])
    plt.xscale('log')
    plt.savefig(output_name)

def find_lr(data, batch_size=1, start_lr=1e-9, end_lr=1):
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(lr=start_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=start_lr)

    model_name = data['training'].origin+'_2_'+data['training'].target + '_generator'
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, model_name)
    if(not os.path.isdir(checkpoint_prefix)):
        os.makedirs(checkpoint_prefix)

    epoch_size = data['training'].__len__()
    lr_mult = (end_lr / start_lr) ** (1 / epoch_size)

    lrs = []
    losses = {
        'gen_mae': []
    }
    best_losses = {
        'gen_mae': 1e9
    }

    print()
    print("Finding the optimal LR with the following parameters: ")
    print("\tCheckpoints: \t", checkpoint_prefix)
    print("\tEpochs: \t", 1)
    print("\tBatchSize: \t", batch_size)
    print("\tnBatches: \t", epoch_size)
    print()    

    print('Epoch {}/{}'.format(1, 1))
    progbar = tf.keras.utils.Progbar(epoch_size)
    for i in range(epoch_size):
        # Get the data from the DataGenerator
        input_image, target = data['training'].__getitem__(i) 
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate a fake image
            gen_output = generator(input_image, training=True)
            # Train the discriminator
            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)
            # Compute the losses
            gen_mae = tf.reduce_mean(tf.abs(target - gen_output))
            gen_loss = generator_loss(disc_generated_output, gen_mae)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            # Compute the gradients
            generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            # Apply the gradients
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
            # Convert losses to numpy 
            gen_mae = gen_mae.numpy()
            # Update the progress bar
            progbar.add(1, values=[("gen_mae", gen_mae)])
            # On batch end
            lr = tf.keras.backend.get_value(generator_optimizer.lr)
            lrs.append(lr)
            # Update the lr
            lr *= lr_mult
            tf.keras.backend.set_value(generator_optimizer.lr, lr)
            tf.keras.backend.set_value(discriminator_optimizer.lr, lr)
            # Update the losses
            losses['gen_mae'].append(gen_mae)
            # Update the best losses
            if(best_losses['gen_mae'] > gen_mae):
                best_losses['gen_mae'] = gen_mae
            if(gen_mae >= 100*best_losses['gen_mae']):
                break

    plot_loss_findlr(losses['gen_mae'], lrs, os.path.join(checkpoint_prefix, 'LRFinder_gen_mae.tiff'))

    print('Best loss:')
    print('gen_mae =', best_losses['gen_mae'])

def train(data, epochs, batch_size=1, lr=1e-3, epoch_offset=0):
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(lr)

    model_name = data['training'].origin+'_2_'+data['training'].target + '_generator'
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, model_name)
    if(not os.path.isdir(checkpoint_prefix)):
        os.makedirs(checkpoint_prefix)
    else:
        if(os.path.isfile(os.path.join(checkpoint_prefix, 'generator.h5'))):
            generator.load_weights(os.path.join(checkpoint_prefix, 'generator.h5'), by_name=True)
            print('Generator weights restorred from ' + checkpoint_prefix)

    # Get the number of batches in the training set
    epoch_size = data['training'].__len__()
    # val_size = data['validation'].__len__()

    print()
    print("Started training with the following parameters: ")
    print("\tCheckpoints: \t", checkpoint_prefix)
    print("\tEpochs: \t", epochs)
    print("\tgen_lr: \t", lr)
    print("\tBatchSize: \t", batch_size)
    print("\tnBatches: \t", epoch_size)
    print()

    # Precompute the test input and target for validation
    audio_input = load_audio(os.path.join(TEST_AUDIOS_PATH, data['training'].origin+'.wav'))
    mag_input, phase = forward_transform(audio_input)
    mag_input = amplitude_to_db(mag_input)
    test_input = slice_magnitude(mag_input, mag_input.shape[0])
    test_input = (test_input * 2) - 1

    audio_target = load_audio(os.path.join(TEST_AUDIOS_PATH, data['training'].target+'.wav'))
    mag_target, _ = forward_transform(audio_target)
    mag_target = amplitude_to_db(mag_target)
    test_target = slice_magnitude(mag_target, mag_target.shape[0])
    test_target = (test_target * 2) - 1

    gen_mae_list, gen_mae_val_list  = [], []
    for epoch in range(epochs):
        gen_mae_total, gen_mae_val_total = 0, 0
        print('Epoch {}/{}'.format((epoch+1)+epoch_offset, epochs+epoch_offset))
        progbar = tf.keras.utils.Progbar(epoch_size)
        for i in range(epoch_size):
            input_image, target = data['training'].__getitem__(i) 
            with tf.GradientTape() as gen_tape:
                # Generate a fake image
                gen_output = generator(input_image, training=True)
                # Compute the losses
                gen_mae = tf.reduce_mean(tf.abs(target - gen_output)) # Timbre transfer
                # gen_mae = tf.reduce_mean(tf.abs(input_image - gen_output)) # Autoencoder
                # Compute the gradients
                generator_gradients = gen_tape.gradient(gen_mae,generator.trainable_variables)        
                # Apply the gradients
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                # Update the progress bar
                gen_mae = gen_mae.numpy()
                gen_mae_total += gen_mae
                progbar.add(1, values=[("gen_mae", gen_mae)])
        
        gen_mae_total /= epoch_size
        gen_mae_list.append(gen_mae_total)
        gen_mae_val_list.append(gen_mae_val_total)
        
        history = pd.DataFrame({
                                    'gen_mae': gen_mae_list, 
                                    'gen_mae_val': gen_mae_val_list
                                })
        write_csv(history, os.path.join(checkpoint_prefix, 'history.csv'))
        
        epoch_output = os.path.join(OUTPUT_PATH, model_name, str((epoch+1)+epoch_offset).zfill(3))
        init_directory(epoch_output)
        # Generate audios and save spectrograms for the entire audios
        prediction = generator(test_input, training=False)
        prediction = (prediction + 1) / 2
        generate_images(prediction, (test_input + 1) / 2, (test_target + 1) / 2, os.path.join(epoch_output, 'spectrogram'))
        generate_audio(prediction, phase, os.path.join(epoch_output, 'audio.wav'))
        print('Epoch outputs saved in ' + epoch_output)

        # Save the weights
        generator.save_weights(os.path.join(checkpoint_prefix, 'generator.h5'))
        print('Weights saved in ' + checkpoint_prefix)

        # Callback at the end of the epoch for the DataGenerator
        data['training'].on_epoch_end()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=False, default='0')
    ap.add_argument('--epochs', required=False, default=100)
    ap.add_argument('--epoch_offset', required=False, default=0)
    ap.add_argument('--batch_size', required=False, default=1)
    ap.add_argument('--lr', required=False, default=5e-6)
    ap.add_argument('--base_path', required=False, default=DATASET_PATH)
    ap.add_argument('--origin', required=False, default='keyboard_acoustic')
    ap.add_argument('--target', required=False, default='guitar_acoustic')
    ap.add_argument('--validation_split', required=False, default=0.9)
    ap.add_argument('--findlr', required=False, default=False)
    args = ap.parse_args()

    # Select which GPU to use and enable tf.foat16
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # tf.keras.backend.set_floatx('float32')

    data = {
        'training': DataGenerator(origin=args.origin, 
                                target=args.target,
                                base_path=args.base_path,
                                batch_size=int(args.batch_size),
                                img_dim=IMG_DIM,
                                validation_split=float(args.validation_split),
                                is_training=True,
                                scale_factor=1),

        'validation': DataGenerator(origin=args.origin, 
                                target=args.target,
                                base_path=args.base_path,
                                batch_size=int(args.batch_size),
                                img_dim=IMG_DIM,
                                validation_split=float(args.validation_split),
                                is_training=False,
                                scale_factor=1,
                                shuffle=False)
    }
    if(args.findlr):
        find_lr(data, int(args.batch_size))
    else:
        train(data, int(args.epochs), int(args.batch_size), float(args.lr), int(args.epoch_offset))
