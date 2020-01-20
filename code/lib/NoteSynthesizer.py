import argparse
import os
import sys

import numpy as np

import pretty_midi
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav

class NoteSynthesizer():
    def __init__(self, dataset_path, sr=44100, transpose=0, leg_stac=.9, velocities=np.arange(0,128), preset=0, preload=True, verbose=1):
        self.dataset_path = dataset_path
        self.sr = sr
        self.transpose = transpose
        self.leg_stac = leg_stac
        self.velocities = velocities
        self.preset = preset

        self.verbose = verbose

        self.preload = preload

    def _get_note_name(self, note, velocity, instrument, source_type, preset=None):
        preset = preset if(preset is not None) else self.preset
        return "%s_%s_%s-%s-%s.wav" % (instrument, source_type, str(preset).zfill(3), str(note).zfill(3), str(velocity).zfill(3))    

    def _quantize(self, value, quantized_values):
        diff = np.array([np.abs(q - value) for q in quantized_values])
        return quantized_values[diff.argmin()]

    def preload_notes(self, instrument, source_type, preset=None):
        preset = preset if(preset is not None) else self.preset
        if(self.verbose):
            print("Preloading notes for " + instrument + "_" + source_type + "_" + str(preset).zfill(3))
        self.notes = {}
        for n in range(22, 108):
            for v in self.velocities:
                note_name = self._get_note_name(n, v, instrument, source_type, preset)
                try:
                    audio, _ = librosa.load(os.path.join(self.dataset_path, note_name), sr=self.sr)
                except:
                    audio = None
                self.notes[note_name] = audio
        print("Notes loaded")

    def _read_midi(self, filename):
        midi_data = pretty_midi.PrettyMIDI(filename)
        end_time = midi_data.get_end_time()
        
        sequence = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start < end_time:
                    note.velocity = self._quantize(note.velocity, self.velocities)
                    sequence.append((note.pitch, note.velocity, note.start/end_time, note.end/end_time))
        return sequence, end_time

    def _render_note(self, note_filename, duration, velocity):
        try:
            if(self.preload):
                note = self.notes[note_filename]
            else:
                note, _ = librosa.load(note_filename)
            decay_ind = int(self.leg_stac*duration)
            envelope = np.exp(-np.arange(len(note)-decay_ind)/3000.)
            note[decay_ind:] = np.multiply(note[decay_ind:],envelope)
        except:
            if(self.verbose):
                print('Note not fonund', note_filename)
            note = np.zeros(duration)
        return note[:duration]

    def render_sequence(self, sequence, instrument='guitar', source_type='acoustic', preset=None, playback_speed=1, duration_scale=1, transpose=0):
        preset = preset if(preset is not None) else self.preset
        transpose = transpose if(transpose is not None) else self.transpose

        seq, end_time = self._read_midi(sequence)
        total_length = int(end_time * self.sr / playback_speed)
        data = np.zeros(total_length)
        
        for note, velocity, note_start, note_end in seq:
            start_sample = int(note_start * total_length)
            end_sample = int(note_end * total_length)
            duration = end_sample - start_sample

            if(duration_scale != 1):
                duration = int(duration * duration_scale)
                end_sample = start_sample + duration
            
            if(self.preload):
                note_filename = self._get_note_name(
                                                        note=note+transpose, 
                                                        velocity=velocity, 
                                                        instrument=instrument, 
                                                        source_type=source_type,
                                                        preset=preset
                                                    )
            else:
                note_filename = os.path.join(self.dataset_path, self._get_note_name(
                                                                            note=note+transpose, 
                                                                            velocity=velocity, 
                                                                            instrument=instrument, 
                                                                            source_type=source_type,
                                                                            preset=preset
                                                                        ))
            note = self._render_note(note_filename, duration, velocity)

            if(end_sample <= len(data) and duration == len(note)):
                data[start_sample:end_sample] += note
            elif(duration > len(note) and end_sample <= len(data)):
                data[start_sample:start_sample+len(note)] += note
            # elif(end_sample > len(data)):
            #     data[start_sample:] = note[0:len(data)-start_sample]

        norm_factor = np.max(np.abs(data)) 
        if(norm_factor > 0):
            data /= norm_factor
        return data, self.sr 

if __name__ == "__main__":
    NSYNTH_SAMPLE_RATE = 16000
    NSYNTH_VELOCITIES = [25, 50, 100, 127]

    ap = argparse.ArgumentParser()
    if len(sys.argv) > -1:
        ap.add_argument('--db', required=True, help="Path to the NSynth audios folder. (ex: /NSynth/nsynth-train/audios)")
        ap.add_argument('--seq', required=True, help="MIDI file (.mid) to be rendered")
        ap.add_argument('--output', required=True, help="Output filename")
        ap.add_argument('--sr', required=False, default=NSYNTH_SAMPLE_RATE, help="Sample rate of the output (default: 16000, typical for professional audio: 44100, 48000)")
        ap.add_argument('--instrument', required=False, default="guitar", help="Name of the NSynth instrument. (default: 'guitar')")
        ap.add_argument('--source_type', required=False, default="acoustic", help="Source type of the NSynth instrument (default: 'acoustic')")
        ap.add_argument('--preset', required=False, default=0, help="Preset of the NSynth instrument (default: 0)")
        ap.add_argument('--transpose', required=False, default=0, help="Transpose the MIDI sequence by a number of semitones")
        ap.add_argument('--playback_speed', required=False, default=1, help="Multiply the sequence length by a scalar (default: 1")
        ap.add_argument('--duration_scale', required=False, default=1, help="Multiply the note durations by a scalar. (default: 1)")
        ap.add_argument('--preload', required=False, default=True, help="Load all notes in memory before rendering for better performance (at least 1 GB of RAM is required)")
    args = vars(ap.parse_args())

    assert os.path.isdir(args['db']), 'Dataset not found in ' + args['db']
    assert os.path.isfile(args['seq']), 'File ' + args['seq'] + ' not found.'

    synth = NoteSynthesizer(
                                args['db'], 
                                sr=NSYNTH_SAMPLE_RATE, 
                                velocities=NSYNTH_VELOCITIES, 
                                preload=args['preload']
                            )    
    if(args['preload']):
        synth.preload_notes(args['instrument'], args['source_type'], int(args['preset']))

    y, _ = synth.render_sequence(
                                    sequence=args['seq'], 
                                    instrument=args['instrument'], 
                                    source_type=args['source_type'], 
                                    preset=int(args['preset']),
                                    transpose=int(args['transpose']),
                                    playback_speed=float(args['playback_speed']),
                                    duration_scale=float(args['duration_scale'])
                                )

    if(int(args['sr']) != NSYNTH_SAMPLE_RATE):
        y = librosa.core.resample(y, NSYNTH_SAMPLE_RATE, int(args['sr']))
    
    write_wav(args['output'], int(args['sr']), np.array(32000.*y, np.short))