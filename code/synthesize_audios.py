import argparse
import os
import subprocess
import sys

import numpy as np
import pandas

import librosa
from config import (AUDIOS_DIRECTORY, DEFAULT_SAMPLING_RATE, MIDI_DIRECTORY,
                    NSYNTH_PATH)
from data import init_directory, list_files, write_audio
from lib.NoteSynthesizer import NoteSynthesizer

NSYNTH_VELOCITIES = [25, 50, 100, 127]
NSYNTH_SAMPLE_RATE = 16000
SAMPLE_RATE = DEFAULT_SAMPLING_RATE

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsynth_path', required=False, default=os.path.join('..', 'data', 'nsynth', 'train', 'audio'))
    ap.add_argument('--midi_path', required=False, default=os.path.join('..', 'data', 'midi', 'Classical_Music_MIDI'))
    ap.add_argument('--audios_path', required=False, default=os.path.join('..', 'data', 'audios', 'Classical_Music_MIDI'))
    ap.add_argument('--playback_speed', required=False, default=1)
    ap.add_argument('--duration_rate', required=False, default=4)
    ap.add_argument('--transpose', required=False, default=0)
    args = ap.parse_args()

    MIDI_DIRECTORY = args.midi_path
    AUDIOS_DIRECTORY = os.path.join(args.audios_path, 'P'+str(args.playback_speed).zfill(1)+'D'+str(args.duration_rate).zfill(1)+'T'+str(args.transpose).zfill(2))

    instruments = [
        {'name': 'guitar', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'keyboard', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'string', 'source_type': 'acoustic', 'preset': 0},
        {'name': 'synth_lead', 'source_type': 'synthetic', 'preset': 0}
    ]
    
    midifiles = list_files(MIDI_DIRECTORY, '.mid')
    init_directory(AUDIOS_DIRECTORY)

    print()
    print("Instruments: \t", len(instruments), [instrument['name'] for instrument in instruments])
    print("MIDI files: \t", len(midifiles))
    print()

    for instrument in instruments:
        synth = NoteSynthesizer(
                                    dataset_path=NSYNTH_PATH, 
                                    sr=NSYNTH_SAMPLE_RATE, 
                                    velocities=NSYNTH_VELOCITIES, 
                                    transpose=float(args.transpose)
                                )
        synth.preload_notes(instrument=instrument['name'], source_type=instrument['source_type'])
        
        instrument_folder = instrument['name']+'_'+instrument['source_type']
        init_directory(os.path.join(AUDIOS_DIRECTORY, instrument_folder))
    
        for mid in midifiles:
            _, seq_name = os.path.split(mid)
            output_name = os.path.join(AUDIOS_DIRECTORY, instrument_folder, os.path.splitext(seq_name)[0]+'.wav')

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
                                                    duration_scale=float(args.duration_scale),
                                                )
                try:
                    if(SAMPLE_RATE != NSYNTH_SAMPLE_RATE):
                        audio = librosa.core.resample(audio, NSYNTH_SAMPLE_RATE, SAMPLE_RATE)
                    write_audio(output_name, audio, SAMPLE_RATE)
                except:
                    print('Could not export audio', output_name)