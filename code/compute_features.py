import argparse
import os

import numpy as np

from data import (amplitude_to_db, forward_transform, init_directory,
                  load_audio, slice_magnitude)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--audios_path', required=True)
    ap.add_argument('--features_path', required=True)
    args = ap.parse_args()

    assert os.path.isdir(args.audios_path), 'Audios not found'

    for instrument in os.listdir(args.audios_path):
        print(instrument)
        audios_dir = os.path.join(args.audios_path, instrument)
        features_dir = os.path.join(args.features_path, instrument)
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