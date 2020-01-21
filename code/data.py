import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np

import librosa
import tensorflow as tf


def files_within(directory_path, pattern="*"):
    for dirpath, _, filenames in os.walk(directory_path):
        for file_name in fnmatch.filter(filenames, pattern):
            yield os.path.join(dirpath, file_name)

def init_directory(directory):
    if(not os.path.isdir(directory)):
        os.makedirs(directory)

def slice_first_dim(array, slice_size):
    n_sections = int(np.floor(array.shape[1]/slice_size))
    has_last_mag = n_sections*slice_size < array.shape[1]

    last_mag = np.zeros(shape=(1, array.shape[0], slice_size, array.shape[2]))
    last_mag[:,:,:array.shape[1]-(n_sections*slice_size),:] = array[:,n_sections*int(slice_size):,:]
    
    if(n_sections > 0):
        array = np.expand_dims(array, axis=0)
        sliced = np.split(array[:,:,0:n_sections*slice_size,:], n_sections, axis=2)
        sliced = np.concatenate(sliced, axis=0)
        if(has_last_mag): # Check for reminder
            sliced = np.concatenate([sliced, last_mag], axis=0)
    else:
        sliced = last_mag
    return sliced

def slice_magnitude(mag, slice_size):
    magnitudes = np.stack([mag], axis=2)
    return slice_first_dim(magnitudes, slice_size)

def join_magnitude_slices(mag_sliced, target_shape):
    mag = np.zeros((mag_sliced.shape[1], mag_sliced.shape[0]*mag_sliced.shape[2]))
    for i in range(mag_sliced.shape[0]):
        mag[:,(i)*mag_sliced.shape[2]:(i+1)*mag_sliced.shape[2]] = mag_sliced[i,:,:,0]
    mag = mag[0:target_shape[0], 0:target_shape[1]]
    return mag

def amplitude_to_db(mag, amin=1/(2**16), normalize=True):
    mag_db = 20*np.log1p(mag/amin)
    if(normalize):
        mag_db /= 20*np.log1p(1/amin)
    return mag_db

def db_to_amplitude(mag_db, amin=1/(2**16), normalize=True):
    if(normalize):
        mag_db *= 20*np.log1p(1/amin)
    return amin*np.expm1(mag_db/20)

def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0],0:mag.shape[1]] = mag
    return rec

def remove_hf(mag):
    return mag[0:int(mag.shape[0]/2), :]

def forward_transform(audio, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft/2), window=window)
    mag, phase = np.abs(S), np.angle(S)
    if(crop_hf):
        mag = remove_hf(mag)
    if(normalize):
        mag = 2 * mag / np.sum(window)
    return mag, phase

def inverse_transform(mag, phase, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    if(normalize):
        mag = mag * np.sum(np.hanning(nfft)) / 2
    if(crop_hf):
        mag = add_hf(mag, target_shape=(phase.shape[0], mag.shape[1]))
    R = mag * np.exp(1j*phase)
    audio = librosa.istft(R, hop_length=int(nfft/2), window=window)
    return audio

def snr(original, reconstruction):
    signal_rms = np.sqrt(np.sum(original**2))
    noise_rms = np.sqrt(np.sum((original - reconstruction)**2))
    return 10.*np.log10(signal_rms/noise_rms)

def load_audio(filename, sr=44100):
    return librosa.core.load(filename, sr=sr)[0]

def write_audio(filename, audio, sr=44100):
    librosa.output.write_wav(filename, audio, sr, norm=True)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, origin, target, base_path, batch_size=1, img_dim=(256,256,1), validation_split=0.9, is_training=True, scale_factor=1.0, shuffle=True):
        self.img_dim = img_dim
        self.batch_size = batch_size
        
        self.validation_split = validation_split
        self.is_training = is_training
        self.scale_factor = scale_factor

        self.base_path = base_path if(type(base_path) is list) else [base_path]

        self.origin = origin
        self.target = target
        self.filenames = self.__get_filenames()
        assert len(self.filenames) > 0, 'Filenames is empty' 

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]         # Generate indexes of the batch
        batch = self.__data_generation(filenames)                                           # Generate data
        return batch

    def get_empty_batch(self):
        batch = np.zeros((self.batch_size, *self.img_dim))
        return batch, batch

    def get_random_batch(self):
        random_idx = np.random.randint(self.__len__())
        return self.__getitem__(random_idx)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples'                                  
        x = np.empty((self.batch_size, *self.img_dim))
        y = np.empty((self.batch_size, *self.img_dim))
        # Generate data
        for i, filename in enumerate(filenames):
            x[i,] = np.load(filename)
            y[i,] = np.load(filename.replace(self.origin, self.target)) 
            if(self.scale_factor != 1):
                x[i,] *= self.scale_factor
                y[i,] *= self.scale_factor
            
            # Now images should be scaled in the range [0,1]. Make them [-1,1]
            x[i,] = x[i,] * 2 - 1
            y[i,] = y[i,] * 2 - 1
        return x,y

    def __get_filenames(self):
        origin_filenames, target_filenames = [],[]
        for base_path in self.base_path:
            origin_temp = [os.path.join(base_path, self.origin, f) for f in os.listdir(os.path.join(base_path, self.origin))]
            target_temp = [os.path.join(base_path, self.target, f) for f in os.listdir(os.path.join(base_path, self.target))]
            if(self.is_training):
                origin_temp = origin_temp[0:int(self.validation_split*len(origin_temp))]
                target_temp = target_temp[0:int(self.validation_split*len(target_temp))]
            else:
                origin_temp = origin_temp[int(self.validation_split*len(origin_temp)):]
                target_temp = target_temp[int(self.validation_split*len(target_temp)):]
            origin_filenames += origin_temp
            target_filenames += target_temp
        if(len(origin_filenames) == len(target_filenames)):
            return origin_filenames
        return []

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.filenames)

class DataGeneratorMultiTarget(tf.keras.utils.Sequence):
    def __init__(self, origin, target, base_path, batch_size=1, img_dim=(256,256,1), validation_split=0.9, is_training=True, scale_factor=1.0, shuffle=True):
        self.img_dim = img_dim
        self.batch_size = batch_size
        
        self.validation_split = validation_split
        self.is_training = is_training
        self.scale_factor = scale_factor

        self.base_path = base_path if(type(base_path) is list) else [base_path]

        self.origin = origin
        self.target = target
        self.filenames = self.__get_filenames()
        assert len(self.filenames) > 0, 'Filenames is empty' 

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'        
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]                 # Generate indexes of the batch
        batch = self.__data_generation(filenames)                      # Generate data
        return batch

    def get_empty_batch(self):
        x = np.zeros((self.batch_size, self.img_dim[0], self.img_dim[1], 2))
        y = np.zeros((self.batch_size, *self.img_dim))
        return x,y 

    def get_random_batch(self):
        random_idx = np.random.randint(self.__len__())
        return self.__getitem__(random_idx)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples'                                  
        x = np.empty((self.batch_size, self.img_dim[0], self.img_dim[1], 2))
        y = np.empty((self.batch_size, *self.img_dim))
        # Generate data
        for i, filename in enumerate(filenames):
            style = np.random.choice(self.filenames)['name']
            x[i,:,:,0:1] = np.load(filename['name'])
            x[i,:,:,1:2] = np.load(style.replace(self.origin, filename['target']))
            y[i,] = np.load(filename['name'].replace(self.origin, filename['target'])) 
            if(self.scale_factor != 1):
                x[i,] *= self.scale_factor
                y[i,] *= self.scale_factor
            # Now images should be scaled in the range [0,1]. Make them [-1,1]
            x[i,] = x[i,] * 2 - 1
            y[i,] = y[i,] * 2 - 1
        return x,y

    def __get_filenames(self):
        origin_filenames = []
        for base_path in self.base_path:
            origin_temp = [os.path.join(base_path, self.origin, f) for f in os.listdir(os.path.join(base_path, self.origin))]
            if(self.is_training):
                origin_temp = origin_temp[0:int(self.validation_split*len(origin_temp))]
            else:
                origin_temp = origin_temp[int(self.validation_split*len(origin_temp)):]
            origin_filenames += origin_temp
        
        filenames = []
        for f in origin_filenames:
            for t in self.target:
                filenames.append({'name': f, 'target': t})     
        return filenames

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.filenames)

class DataGeneratorAny2Any(tf.keras.utils.Sequence):
    def __init__(self, base_path, batch_size=1, img_dim=(256,256,1), validation_split=0.9, is_training=True, scale_factor=1.0, shuffle=True):
        self.img_dim = img_dim
        self.batch_size = batch_size
        
        self.validation_split = validation_split
        self.is_training = is_training
        self.scale_factor = scale_factor

        self.base_path = base_path if(type(base_path) is list) else [base_path]

        self.instruments = self.__get_instruments()
        self.filenames = self.__get_filenames()
        assert len(self.filenames) > 0, 'Filenames is empty' 

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'        
        batch_filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]                   # Generate indexes of the batch
        batch = self.__data_generation(batch_filenames)                                                     # Generate data
        return batch

    def get_empty_batch(self):
        x = np.zeros((self.batch_size, self.img_dim[0], self.img_dim[1], 2))
        y = np.zeros((self.batch_size, *self.img_dim))
        return x,y 

    def get_random_batch(self):
        random_idx = np.random.randint(self.__len__())
        return self.__getitem__(random_idx)

    def __get_instruments(self):
        instruments = [ f for f in os.listdir(self.base_path) if os.path.isdir(f) ]
        return instruments

    def __get_style(self, target):
        style = np.random.choice(self.filenames)['name']        # From outside the batch (pick from all filenames)
        style.replace('instrument_placeholder', target)         # Make it match the target
        return style 

    def __data_generation(self, batch_filenames):
        'Generates data containing batch_size samples'                                  
        x = np.empty((self.batch_size, self.img_dim[0], self.img_dim[1], 2))
        y = np.empty((self.batch_size, *self.img_dim))
        # Generate data
        for i, filename in enumerate(batch_filenames):
            style = self.__get_style(filename['target'])                                           
            original = filename['name'].replace('instrument_placeholder', filename['origin'])
            output = filename['name'].replace('instrument_placeholder', filename['target'])

            x[i,:,:,0:1] = np.load(original)
            x[i,:,:,1:2] = np.load(style)
            y[i,] = np.load(output) 

            if(self.scale_factor != 1):
                x[i,] *= self.scale_factor
                y[i,] *= self.scale_factor
            # Now images should be scaled in the range [0,1]. Make them [-1,1]
            x[i,] = x[i,] * 2 - 1
            y[i,] = y[i,] * 2 - 1
        return x,y

    def __get_filenames(self):
        origin_filenames = []
        for base_path in self.base_path:
            origin_temp = []
            origin_temp.append(*[os.path.join(base_path, 'instrument_placeholder', f) for f in os.listdir(os.path.join(base_path, self.instruments[0]))]) # Arbitrary choice [0]
            if(self.is_training):
                origin_temp = origin_temp[0:int(self.validation_split*len(origin_temp))]
            else:
                origin_temp = origin_temp[int(self.validation_split*len(origin_temp)):]
            origin_filenames += origin_temp
        
        filenames = []
        for f in origin_filenames:
            for o in self.instruments:
                for t in self.instruments:
                    if(o != t):
                        filenames.append({'name': f, 'origin': o, 'target': t})     
        return filenames

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.filenames)
