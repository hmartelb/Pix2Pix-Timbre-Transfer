import os, sys
import argparse
import numpy as np
import shutil

def init_directory(path):
    if(not os.path.isdir(path)):
        os.makedirs(path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--reduced_path', required=True)
    # ap.add_argument('--decimation_factor', required=False, default=0.1)
    args = ap.parse_args()

    assert os.path.isdir(args.dataset_path), "Dataset path not found!"
    init_directory(args.reduced_path)
    
    first_folder = os.listdir(args.dataset_path)[0]
    filenames = [f for f in os.listdir(os.path.join(args.dataset_path, first_folder)) if(os.path.isfile(os.path.join(args.dataset_path, first_folder, f)))]
    print('Total filenames = ', len(filenames))

    assert len(filenames) > 0, "No filenames found!"

    # Select randomly
    decimation_factor = float(input("Select the decimation factor: "))
    if(decimation_factor > 0 and decimation_factor < 1):
        filenames = np.random.choice(filenames, int(len(filenames)*decimation_factor))
    print('Random selection. Size after = ', len(filenames))

    for folder in os.listdir(args.dataset_path):
        origin_path = os.path.join(args.dataset_path, folder)
        destination_path = os.path.join(args.reduced_path, folder)
        init_directory(destination_path)

        for file in filenames:
            print('Copying '+ file + ' from ' + origin_path + ' to ' + destination_path)
            shutil.copyfile(
                                src=os.path.join(origin_path, file), 
                                dst=os.path.join(destination_path, file)
                            )