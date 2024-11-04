import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import tensorflow as tf
#from tensorflow.keras import models

def show_fig(label, xaxis_min, xaxis_max, xaxis_step, Intensity, outputPath, flag):
  if flag == 'tth':
    xlabel = '2θ [deg.]'
  elif flag == 'Qrlu':
    xlabel = 'Q [r.l.u.]'
  
  if type(label) == int:
    if label == 0.0:
      title = 'Others'
    elif label == 1.0:
      title = 'QC'
  else:
    if type(label) == str:
        title = label
    else:
       title = 'label: '+str(label)
  xaxis = np.arange(xaxis_min, xaxis_max, xaxis_step)
  fig = plt.figure(figsize=(9, 6))
  fig, ax = plt.subplots()
  ax.tick_params(labelleft=False)
  plt.plot(xaxis, Intensity, zorder=1)
  plt.title(title)
  #plt.legend()
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.xlabel(xlabel)
  plt.ylabel('Intensity [a.u.]')
  plt.savefig(outputPath)
  return


def save_training_data(label, xaxis_min, xaxis_max, xaxis_step, x_train, y_train, n_data, output_path, flag):
  if flag == 'tth':
    xlabel = '2θ [deg.]'
  elif flag == 'Qrlu':
    xlabel = 'Q [r.l.u.]'

  xaxis = np.arange(xaxis_min, xaxis_max, xaxis_step)

  #n_data = 6 # 表示するデータ数
  row = 2 # 行数
  col = 3 # 列数
  fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(8,6))

  fig.suptitle("MNIST data-set")
  for i in enumerate(range(len(x_train[:n_data]))):
      _r = i//col
      _c = i%col
      ax[_r,_c].set_title(y_train[i], fontsize=16, color='white')
      ax[_r,_c].show(x_train[i]) # 画像を表示

  return

def save_16plots(intensity_list, outputDir):
  if len(intensity_list)!=16:
     print('Error: Not enough data num., got %s data, expect 16 data'%(str(len(intensity_list))))
     return
  a = 0
  fig, axs = plt.subplots(4, 4, figsize=(20,10))
  plt.ylim(0, 1)
  for i in range(4):
    for j in range(4):
      # if i==0 and j == 0:
      #   continue
      tths = np.arange(20,80,0.01)
      plt.ylim(0, 1)
      axs[i][j].plot(tths, intensity_list[a])
      a+=1
  try:
    plt.savefig(outputDir+'/PXRD.png')
  except:
    pass
  return


def Get_GradCAM(input_model, layer_name, preprocessed_input):
  import matplotlib.pyplot as plt
  import numpy as np
  import pylab
  import math
  from scipy import integrate
  import time
  import cProfile
  import random
  import keras
  import tensorflow as tf
  from tensorflow.keras.layers import Dense, Flatten, Conv1D ,Dropout, MaxPooling1D ,InputLayer, Reshape
  from keras.layers.pooling import GlobalMaxPooling1D
  from tensorflow.keras import Model
  from keras import regularizers


  import pandas as pd
  import numpy as np
  import cv2
  from keras import backend as K
  from keras.models import load_model

  K.set_learning_phase(1) #set learning phase


  from tensorflow.keras import models
  import tensorflow as tf

  import pandas as pd
  import seaborn as sns
  from mylib import save_fig
  pred = input_model.predict(preprocessed_input)
  print('pred: ', pred)

  #x=x_test_
  #input_model=mode1
  #layer_name='conv1d_5'

  #preprocessed_input = x

  grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(preprocessed_input)
      class_idx = np.argmax(predictions[0])
      loss = predictions[:, class_idx]

  # 勾配を計算
  output = conv_outputs[0]
  grads = tape.gradient(loss, conv_outputs)[0]

  gate_f = tf.cast(output > 0, 'float64')
  gate_r = tf.cast(grads > 0, 'float64')

  guided_grads = gate_f * gate_r * grads

  # 重みを平均化して、レイヤーの出力に乗じる
  weights = np.mean(guided_grads, axis=0)
  cam = np.dot(output, weights)


  dic_={}
  count=0
  min_=1000000
  max_=-10000000
  for i in range(len(cam)):
      #for j in range(18):
          if cam[i]>max_:
              max_=cam[i]
          elif cam[i]<min_:
              min_=cam[i]
          else:
              uri=1
          dic_[count]=[cam[i]]
          count+=1
  data=pd.DataFrame(dic_)
  #g=sns.heatmap(data, cmap='bwr')

  if pred[0][1]>0.5:
      label = 1.0
  else:
      label = 0.0

  xlabel = '2θ [deg.]'
  outputPath='./models/junk/'
  if type(label) == int:
    if label == 0.0:
      title = 'Others'
    elif label == 1.0:
      title = 'QC'
  else:
    if type(label) == str:
        title = label
    else:
       title = 'label: '+str(label)
  xaxis = np.arange(20.0, 80.0, 0.01)
  fig = plt.figure(figsize=(20, 4))
  #fig, ax = plt.subplots()
  #ax.tick_params(labelleft=False)
  plt.plot(xaxis, preprocessed_input[0], zorder=1)
  plt.title(title)
  plt.rcParams["figure.figsize"] = (20, 4)
  #plt.legend()
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.xlabel(xlabel)
  plt.ylabel('Intensity [a.u.]')
  plt.savefig(outputPath)
  #show_fig(label, 20.0, 80.0, 0.01, preprocessed_input[0], outputPath='./models/junk/', flag='tth')

  fig = plt.figure(figsize=(18,2))
  g = sns.heatmap(data, cmap='Greys', cbar=False)
  fig.savefig('./models/junk/CAMs_test.png',format = 'png', dpi=600,  bbox_inches="tight")
  """
  def Lorenz(p,x):
      return ((p[0]*p[1]**2/((x-p[2])**2+p[1]**2)))
  #39.9761,37.965
  tths=np.arange(20,80,0.01)
  theta=37.965
  y+=Lorenz([0.3,0.05,theta],tths)
  pylab.plot(tths,y,lw=0.5)
  yy=y.reshape(6000,1)
  tf.keras.backend.set_floatx('float64')
  yy = yy[..., tf.newaxis]
  yy=yy.reshape(1,6000,1)
  model.predict(yy)
  """
  return