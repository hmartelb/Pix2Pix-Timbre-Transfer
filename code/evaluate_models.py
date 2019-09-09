import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import librosa
from config import DEFAULT_SAMPLING_RATE
from data import load_audio, snr

def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def compute_snr(dataset, predictions):
    assert len(dataset) == len(predictions), 'Dataset and predictions do not match'
    result = np.empty(shape=[len(dataset)])
    for i in range(len(dataset)):
        original_audio = librosa.load(dataset[i], sr=DEFAULT_SAMPLING_RATE)
        reconstruction_audio = librosa.load(predictions[i], sr=DEFAULT_SAMPLING_RATE)
        result[i] = snr(original_audio, reconstruction_audio)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--predictions', required=True)
    args = ap.parse_args()

    dataset_filenames = list_files(args.dataset)
    prediction_directories = os.listdir(args.prediction)

    result = pd.DataFrame()
    for i in range(len(prediction_directories)):
        prediction_filenames = list_files(os.path.join(args.predictions, prediction_directories[i]))
        column = pd.DataFrame({
                                prediction_directories[i]: compute_snr(dataset_filenames, prediction_filenames)
                            })
        result = pd.concat([result, column], axis=1)

    