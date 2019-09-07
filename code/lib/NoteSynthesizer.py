import os
import sys

import numpy as np

import pretty_midi
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav

class NoteSynthesizer():
    def __init__(self, dataset_path, sr=44100, transpose=0, leg_stac=.9, velocities=np.arange(0,128), preset=0, verbose=False, preload=True):
        self.dataset_path = dataset_path
        self.sr = sr
        self.transpose = transpose
        self.leg_stac = leg_stac
        self.velocities = velocities
        self.preset = preset

        self.preload = preload
        self.verbose = verbose

    def _get_note_name(self, note, velocity, instrument, source_type, preset=None):
        preset = preset if(preset is not None) else self.preset
        return "%s_%s_%s-%s-%s.wav" % (instrument, source_type, str(preset).zfill(3), str(note).zfill(3), str(velocity).zfill(3))    

    def _quantize(self, value, quantized_values):
        diff = np.array([np.abs(q - value) for q in quantized_values])
        return quantized_values[diff.argmin()]

    def preload_notes(self, instrument, source_type, preset=None):
        preset = preset if(preset is not None) else self.preset
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
                print('Note not found, rendering silence')
            note = np.zeros(duration)
        return note[:duration]

    def render_sequence(self, sequence, instrument, source_type='acoustic', preset=None, playback_speed=1, duration_scale=1):
        preset = preset if(preset is not None) else self.preset
        
        seq, end_time = self._read_midi(sequence)
        total_length = int(end_time * self.sr / playback_speed)
        data = np.zeros(total_length)
        
        for note, velocity, note_start, note_end in seq:
            start_sample = int(note_start * total_length)
            end_sample = int(note_end * total_length)
            duration = end_sample - start_sample

            if(duration_scale != 1):
                duration *= duration_scale
                end_sample = start_sample + duration
            
            if(self.preload):
                note_filename = self._get_note_name(
                                                        note=note+self.transpose, 
                                                        velocity=velocity, 
                                                        instrument=instrument, 
                                                        source_type=source_type,
                                                        preset=preset
                                                    )
            else:
                note_filename = os.path.join(self.dataset_path, self._get_note_name(
                                                                            note=note+self.transpose, 
                                                                            velocity=velocity, 
                                                                            instrument=instrument, 
                                                                            source_type=source_type,
                                                                            preset=preset
                                                                        ))
            try:
                data[start_sample:end_sample] += self._render_note(note_filename, duration, velocity)
            except:
                pass

        data /= np.max(np.abs(data)) 
        return data, self.sr