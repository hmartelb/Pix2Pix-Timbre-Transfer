import argparse
import os

import numpy as np

from config import CHECKPOINT_DIR, DEFAULT_SAMPLING_RATE, IMG_DIM, OUTPUT_PATH
from data import (amplitude_to_db, db_to_amplitude, forward_transform,
                  init_directory, inverse_transform, join_magnitude_slices,
                  load_audio, slice_magnitude, write_audio)
from model import Generator


def predict(model, input_filename, output_filename):
    audio = load_audio(input_filename, sr=DEFAULT_SAMPLING_RATE)
    mag, phase = forward_transform(audio)
    mag_db = amplitude_to_db(mag)
    mag_sliced = slice_magnitude(mag_db, IMG_DIM[1])

    prediction = model.predict(mag_sliced)

    mag_db = join_magnitude_slices(prediction, phase.shape)
    mag = db_to_amplitude(mag_db)
    audio_out = inverse_transform(mag, phase)
    write_audio(output_filename, audio_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu', required=False, default='0')
    args = ap.parse_args()

    assert os.path.isfile(args.model), 'Model not found'
    assert os.path.isfile(args.input), 'Input audio not found'
    
    _, ext = os.path.splitext(args.input)
    assert ext in ['.wav', '.mp3', '.ogg'], 'Invalid audio format'

    # Select which GPU to use and enable tf.foat16
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    model = Generator()
    model.load_weights(args.model)
    print('Weights loaded from', args.model)

    base_output_path, _ = os.path.split(args.output)
    init_directory(base_output_path)
    print('Created directory', base_output_path)
    
    predict(model, args.input, args.output)