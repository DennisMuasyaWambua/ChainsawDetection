import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


os.chdir('C:/Users/ADMIN/Desktop/urbanSound/audio')
print(os.getcwd())
print(os.listdir())
metadata = pd.read_csv('C:/Users/ADMIN/Desktop/ChainsawDetection/esc50.csv')
df = pd.DataFrame(metadata)
df.set_index('filename', inplace=True)
for f in df.index:
  rate, signal = wavfile.read('C:/Users/ADMIN/Desktop/ChainsawDetection/audio/'+f)
  df.at[f,'length'] = signal.shape[0]/rate
classes = list(np.unique(df.category))
classes_dist = df.groupby(['category'])['length'].mean()
print(classes_dist)
fig, ax = plt.subplots()
ax.set_title("Class Distribution", y=1.08)
ax.pie(classes_dist, labels=classes_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)
signals = {}
fft = {}
fbank = {}
mfccs = {}
def calc_fft(y, rate):
  n = len(y)
  freq = np.fft.rfftfreq(n, d=1/rate)
  y = abs(np.fft.rfft(y)/n)
  return (y,freq)
# #graphs
def plot_signals(signals):
  fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
  fig.suptitle('Time Series', size=16)
  i=0
  for x in range(2):
    for y in range(5):
      axes[x,y].set_title(list(signals.keys())[i])
      axes[x,y].plot(list(signals.values())[i])
      axes[x,y].get_xaxis().set_visible(False)
      axes[x,y].get_yaxis().set_visible(False)
      i+=1
def plot_fft(fft):
  fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
  fig.suptitle('Time Series', size=16)
  i=0
  for x in range(2):
    for y in range(5):
      data = list(fft.values())[i]
      Y, freq = data[0], data[1]
      axes[x,y].set_title(list(fft.keys())[i])
      axes[x,y].plot(freq,Y)
      axes[x,y].get_xaxis().set_visible(False)
      axes[x,y].get_yaxis().set_visible(False)
      i+=1
def plot_mfccs(mfccs):
  fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
  fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
  i=0
  for x in range(2):
    for y in range(5):
      axes[x,y].set_title(list(mfccs.keys())[i])
      axes[x,y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
      axes[x,y].get_xaxis().set_visible(False)
      axes[x,y].get_yaxis().set_visible(False)
      i+=1
for c in classes:
  wav_file = df[df.category == c].iloc[0,0]
  signal, rate = librosa.load('C:/Users/ADMIN/Desktop/ChainsawDetection/audio/'+wav_file, sr=44100)
  signals[c] = signal
  fft[c]= calc_fft(signal, rate)

  bank = logfbank(signal[:rate],rate, nfilt=26, nfft=1103).T
  fbank[c] = bank
  mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
  mfccs[c]= mel
plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()

plot_mfccs(mfccs)
plt.show()
#
# def feature_extraction(file):
#   audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
#   mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#   mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
#   return mfccs_scaled_features
#
# extracted_features = []
#
# for index_num, row in tqdm(metadata.iterrows()):
#   files = os.path.join(os.getcwd(), str(row['filename']))
#   final_class_labels= row["category"]
#   data = feature_extraction(files)
#   extracted_features.append([data, final_class_labels])
#
# extracted_features_df = pd.DataFrame(extracted_features,columns=['feature', 'class'])
#
# print(extracted_features_df.head())
#
# #converting the data into independent and dependent features
# x = np.array(extracted_features_df['feature'].tolist())
# y = np.array(extracted_features_df['class'].tolist())
#
# print(x.shape)
