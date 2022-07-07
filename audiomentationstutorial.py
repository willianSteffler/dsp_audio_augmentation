import calendar
import datetime
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter,ApplyImpulseResponse,Resample,Trim,Normalize
import os

from helper import _plot_signal_and_augmented_signal

# install pydub for using HighPassFilter
# install audiomentations

amostras=10
folder='normalized/segundo_andar'
saida='augmented'

def get_utc():
    date = datetime.datetime.utcnow()
    return calendar.timegm(date.utctimetuple())

# Raw audio augmentation
augment_raw_audio = Compose(
    [
        # Resample(min_sample_rate=22050, max_sample_rate=22050, p=1),
        # Normalize(1),
        # Trim(),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.25),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ApplyImpulseResponse(ir_path='./impulse_responses/MIT',p=1),
        # HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1)
    ]
)

if __name__ == "__main__":

    folder_name = folder.replace('\\','/')
    if '/' in folder_name:
        folder_name = folder.rsplit('/',1)[1]
    
    output_folder = f'{saida}/{folder_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root,dirs,files in os.walk(folder):

        for file in files:

            signal, sr = librosa.load(f'{root}/{file}')
            
            for v in range(amostras):
                suffix = v
                name = file.rsplit('.',1)[0]
                augmented_signal = augment_raw_audio(signal, sr)
                sf.write(f'{output_folder}/{name}_{suffix}.wav', augmented_signal, sr)
                print(f'generated {output_folder}/{name}_{suffix}.wav')
                # _plot_signal_and_augmented_signal(signal, augmented_signal, sr)