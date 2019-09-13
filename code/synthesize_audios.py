import argparse
import os
import subprocess
import sys

import numpy as np
import pandas

import librosa
from config import DEFAULT_SAMPLING_RATE, NSYNTH_SAMPLE_RATE, NSYNTH_VELOCITIES
from data import files_within, init_directory
from lib.NoteSynthesizer import NoteSynthesizer
from scipy.io.wavfile import write as write_wav

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsynth_path', required=True)
    ap.add_argument('--midi_path', required=True)
    ap.add_argument('--audios_path', required=True)
    ap.add_argument('--playback_speed', required=False, default=1)
    ap.add_argument('--duration_rate', required=False, default=4)
    ap.add_argument('--transpose', required=False, default=0)
    args = ap.parse_args()

    assert os.path.isdir(args.nsynth_path), 'NSynth Dataset not found'
    assert os.path.isdir(args.midi_path), 'MIDI Dataset not found'

    instruments = [
        {'name': 'guitar', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'keyboard', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'string', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'synth_lead', 'source_type': 'synthetic', 'preset': 0}
    ]
    
    midifiles = list(files_within(args.midi_path, '*.mid'))
    init_directory(args.audios_path)

    print()
    print("Instruments: \t", len(instruments), [instrument['name'] for instrument in instruments])
    print("MIDI files: \t", len(midifiles))
    print()

    for instrument in instruments:
        synth = NoteSynthesizer(
                                    dataset_path=args.nsynth_path, 
                                    sr=NSYNTH_SAMPLE_RATE, 
                                    velocities=NSYNTH_VELOCITIES, 
                                    transpose=float(args.transpose)
                                )
        synth.preload_notes(instrument=instrument['name'], source_type=instrument['source_type'])
        
        instrument_folder = instrument['name']+'_'+instrument['source_type']
        init_directory(os.path.join(args.audios_path, instrument_folder))
    
        for mid in midifiles:
            _, seq_name = os.path.split(mid)
            output_name = os.path.join(args.audios_path, instrument_folder, os.path.splitext(seq_name)[0]+'.wav')

            print("Instrument: \t", instrument_folder)
            print("Sequence: \t", mid)
            print("Output: \t", output_name, '\n')

            if(not os.path.isfile(output_name)):
                audio, _ = synth.render_sequence(
                                                    sequence=str(mid),
                                                    instrument=instrument['name'],
                                                    source_type=instrument['source_type'],
                                                    preset=instrument['preset'],
                                                    playback_speed=float(args.playback_speed),
                                                    duration_scale=float(args.duration_rate),
                                                )

                if(DEFAULT_SAMPLING_RATE != NSYNTH_SAMPLE_RATE):
                    audio = librosa.core.resample(audio, NSYNTH_SAMPLE_RATE, DEFAULT_SAMPLING_RATE)
                # write_audio(output_name, audio, DEFAULT_SAMPLING_RATE)
                write_wav(output_name, DEFAULT_SAMPLING_RATE, np.array(32000.*audio, np.short))