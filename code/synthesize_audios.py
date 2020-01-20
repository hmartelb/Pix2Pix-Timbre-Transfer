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
        {'name': 'bass', 'source_type': 'acoustic'},
        {'name': 'bass', 'source_type': 'electronic'},
        {'name': 'bass', 'source_type': 'synthetic'},
        
        {'name': 'brass', 'source_type': 'acoustic'},
        {'name': 'brass', 'source_type': 'electronic'},
        # {'name': 'brass', 'source_type': 'synthetic'},
        
        {'name': 'flute', 'source_type': 'acoustic'},
        {'name': 'flute', 'source_type': 'electronic'},
        {'name': 'flute', 'source_type': 'synthetic'},

        {'name': 'guitar', 'source_type': 'acoustic'},
        {'name': 'guitar', 'source_type': 'electronic'},
        {'name': 'guitar', 'source_type': 'synthetic'},

        {'name': 'keyboard', 'source_type': 'acoustic'},
        {'name': 'keyboard', 'source_type': 'electronic'},
        {'name': 'keyboard', 'source_type': 'synthetic'},

        {'name': 'mallet', 'source_type': 'acoustic'},
        {'name': 'mallet', 'source_type': 'electronic'},
        {'name': 'mallet', 'source_type': 'synthetic'},

        {'name': 'organ', 'source_type': 'acoustic'},
        {'name': 'organ', 'source_type': 'electronic'},
        # {'name': 'organ', 'source_type': 'synthetic'},

        {'name': 'reed', 'source_type': 'acoustic'},
        {'name': 'reed', 'source_type': 'electronic'},
        {'name': 'reed', 'source_type': 'synthetic'},

        {'name': 'string', 'source_type': 'acoustic'},
        {'name': 'string', 'source_type': 'electronic'},
        # {'name': 'string', 'source_type': 'synthetic'},

        # {'name': 'synth_lead', 'source_type': 'synthetic'},
        # {'name': 'synth_lead', 'source_type': 'electronic'},
        {'name': 'synth_lead', 'source_type': 'synthetic'}
    ]
    
    midifiles = list(files_within(args.midi_path, '*.mid'))
    init_directory(args.audios_path)

    print()
    print("Instruments: \t", len(instruments), [instrument['name'] for instrument in instruments])
    print("MIDI files: \t", len(midifiles))
    print()

    #
    # Note qualities according to the NSynth specification:
    # https://magenta.tensorflow.org/datasets/nsynth#note-qualities
    #
    # 0	    bright	        A large amount of high frequency content and strong upper harmonics.
    # 1	    dark	        A distinct lack of high frequency content, giving a muted and bassy sound. Also sometimes described as ‘Warm’.
    # 2	    distortion	    Waveshaping that produces a distinctive crunchy sound and presence of many harmonics. Sometimes paired with non-harmonic noise.
    # 3	    fast_decay	    Amplitude envelope of all harmonics decays substantially before the ‘note-off’ point at 3 seconds.
    # 4	    long_release	Amplitude envelope decays slowly after the ‘note-off’ point, sometimes still present at the end of the sample 4 seconds.
    # 5	    multiphonic	    Presence of overtone frequencies related to more than one fundamental frequency.
    # 6	    nonlinear_env	Modulation of the sound with a distinct envelope behavior different than the monotonic decrease of the note. Can also include filter envelopes as well as dynamic envelopes.
    # 7	    percussive	    A loud non-harmonic sound at note onset.
    # 8	    reverb	        Room acoustics that were not able to be removed from the original sample. 
    #
    # for note_quality in range(0,9): 
    note_quality = 0
    for instrument in instruments:
        synth = NoteSynthesizer(
                                    dataset_path=args.nsynth_path, 
                                    sr=NSYNTH_SAMPLE_RATE, 
                                    velocities=NSYNTH_VELOCITIES, 
                                    transpose=float(args.transpose),
                                    verbose=0
                                )
        synth.preload_notes(instrument=instrument['name'], source_type=instrument['source_type'])
        
        instrument_folder = instrument['name']+'_'+instrument['source_type']+'_'+str(note_quality).zfill(3)
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
                                                    preset=note_quality,
                                                    playback_speed=float(args.playback_speed),
                                                    duration_scale=float(args.duration_rate),
                                                )

                if(DEFAULT_SAMPLING_RATE != NSYNTH_SAMPLE_RATE):
                    audio = librosa.core.resample(audio, NSYNTH_SAMPLE_RATE, DEFAULT_SAMPLING_RATE)
                # write_audio(output_name, audio, DEFAULT_SAMPLING_RATE)
                write_wav(output_name, DEFAULT_SAMPLING_RATE, np.array(32000.*audio, np.short))