import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=False, default='keyboard_acoustic')
    ap.add_argument('--target', required=False, default='guitar_acoustic')
    ap.add_argument('--models_path', required=False, default='models')
    args = ap.parse_args()

    history_name = os.path.join(args.models_path, args.origin+'_2_'+args.target, 'history.csv')
    history = pd.read_csv(history_name)

    min_losses_name = os.path.join(args.models_path, args.origin+'_2_'+args.target, 'min_losses.txt')
    if(not os.path.isfile(min_losses_name)):
        open(min_losses_name, 'w') 
    
    for key in ['gen_mae', 'disc_loss', 'gen_loss']:
        plt.plot(history[key])
        with open(min_losses_name, 'a') as f:
            f.write(key+'_best: '+ str(min(history[key]))+'\n')
            f.write(key+'_last: '+ str(history[key][99])+'\n')
        fig_name = os.path.join(args.models_path, args.origin+'_2_'+args.target, key+'_history.png')
        plt.savefig(fig_name)
        plt.clf()