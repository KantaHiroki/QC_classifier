import codecs
import numpy as np
import tensorflow as tf


def data_compensate(tth_list, Intensity_list, tth_interval):
    tth_len = 6000
    tth_len_ = int((80.0-20.0)/tth_interval)
    step = int(tth_len/tth_len_)
    pattern_list = []
    for i in range(len(tth_list)):
        intensity1 = Intensity_list[i]
        pattern_list.append(intensity1)
        if i==len(tth_list)-1:
            intensity2 = Intensity_list[i]
        else:
            intensity2 = Intensity_list[i+1]
            intensity = (intensity1+intensity2)/step
        pattern_list.append(intensity)
    return pattern_list

def read_pattern_in_file(file_path):
    tth_list = []
    Intensity_list = []
    try:
        # f = open(file_path, 'r', encoding="utf-8")
        with codecs.open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except:
        # f = open(file_path, 'r', encoding="shift-jis")
        with codecs.open(file_path, 'r', encoding="shift-jis") as f:
            lines = f.readlines()
    # try:
    for line in lines:
        if '*' in line:
            continue
        elif '#' in line:
            continue
        line_list = line[:-1].split(' ')
        try:
            tth = float(line_list[0])
        except:
            continue
        if 20 <= tth < 80:
            tth_list.append(tth)
            Intensity = float(line_list[1])
            Intensity_list.append(Intensity)
    if len(Intensity_list)!=6000:
        print('Warning : Not appropriate 2θ step.')
        tth_interval = tth_list[1]-tth_list[0]
        Intensity_list = data_compensate(tth_list, Intensity_list, tth_interval)
    # except:
        # return print('Error : Unknown problem.')
    return Intensity_list

def preprocess_pattern(Intensity_list):
    x_test = np.array([Intensity_list], np.float64)
    x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
    x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
    tf.keras.backend.set_floatx('float64')
    x_test_ = x_test[..., tf.newaxis]
    return x_test_