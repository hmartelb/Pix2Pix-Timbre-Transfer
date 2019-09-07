import os
from data import load_audio, forward_transform, amplitude_to_db, init_directory, slice_magnitude
import numpy as np
import argparse

from config import DATASET_AUDIOS_PATH, DATASET_FEATURES_PATH

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--audios', required=False, default=DATASET_AUDIOS_PATH)
    ap.add_argument('--features', required=False, default=DATASET_FEATURES_PATH)
    args = ap.parse_args()

    for instrument in os.listdir(args.audios):
        print(instrument)
        audios_dir = os.path.join(args.audios, instrument)
        features_dir = os.path.join(args.features, instrument)
        init_directory(features_dir)

        for f in os.listdir(audios_dir):
            name, _ = os.path.splitext(f)
            
            audio = load_audio(os.path.join(audios_dir, f))
            mag, _ = forward_transform(audio)
            mag = amplitude_to_db(mag)
            
            mag_sliced = slice_magnitude(mag, mag.shape[0])
            
            print(name, mag_sliced.shape[0])
            for i in range(mag_sliced.shape[0]):
                out_name = os.path.join(features_dir, name+'_'+str(i).zfill(3)+'.npy')
                if(not os.path.isfile(out_name)):
                    np.save(out_name, mag_sliced[i,:,:,:])