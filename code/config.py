import os

IMG_DIM = (256,256,1)

DEFAULT_SAMPLING_RATE = 44100

# NSYNTH_PATH = os.path.join('..','data', 'audios', 'nsynth')
NSYNTH_VELOCITIES = [25, 50, 100, 127]
NSYNTH_SAMPLE_RATE = 16000

DATASET_AUDIOS_PATH = os.path.join('..','data','audios','Classical_Music_MIDI')
DATASET_FEATURES_PATH = os.path.join('..', 'data', 'features', 'Classical_Music_MIDI')

TEST_AUDIOS_PATH = os.path.join('..','data','audios', 'test')

OUTPUT_PATH = os.path.join('..','data', 'outputs')
CHECKPOINT_DIR = os.path.join('..', 'models')