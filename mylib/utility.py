import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm
import pandas as pd

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


def read_pattern_in_file(file_path, xaxis_min, xaxis_max, xaxis_step):
    tth_list = []
    Intensity_list = []
    try:
        with codecs.open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except:
        with codecs.open(file_path, 'r', encoding="shift-jis") as f:
            lines = f.readlines()
    for line in lines:
        if line[0] == '*':
            continue
        elif line[0]=='#':
            continue
        line_list = line.split()
        try:
            tth = float(line_list[0])
        except:
            continue
        if xaxis_min <= tth < xaxis_max:
            tth_list.append(tth)
            Intensity = float(line_list[1])
            Intensity_list.append(Intensity)
    f.close()

    if len(Intensity_list)<int((xaxis_max-xaxis_min)/xaxis_step):
        tth_interval = tth_list[1]-tth_list[0]
        Intensity_list = data_compensate(tth_list, Intensity_list, tth_interval)
    return Intensity_list


def data_download_from_list(path_exptdir, path_file_list, xaxis_min, xaxis_max, xaxis_step):
    files = {}
    file_list = open(path_file_list, 'r', encoding="utf-8")
    for file_name in file_list.readlines():
        try:
            tth_list = []
            Intensity_list = []
            f = open(path_exptdir+file_name[:-1], 'r', encoding="shift-jis")
            for line in f.readlines():
                if line[0] == '*':
                    continue
                elif line[0]=='#':
                    continue
                line_list = line[:-1].split()
                tth = float(line_list[0])
                if xaxis_min <= tth < xaxis_max:
                    tth_list.append(tth)
                    Intensity = float(line_list[1])
                    Intensity_list.append(Intensity)
            f.close()

            if len(Intensity_list)!=int((xaxis_max-xaxis_min)/xaxis_step):
                print("="*80)
                print('Error! 2 theta step of this data in measurement not suppposed. (%s)'%(file_name))
                print("="*80)
               
            x_test = np.array([Intensity_list], np.float64)
            x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
            x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
            tf.keras.backend.set_floatx('float64')
            x_test_ = x_test[..., tf.newaxis]
            files[file_name] = x_test_
        except:
            print("="*80)
            print('Error! Please check the file.')
            print(file_name)
            print("="*80)

    file_list.close()
    return files

def data_download(path_exptdata, extension, xaxis_min, xaxis_max, xaxis_step):
    files = {}
    extension_list = [extension, extension.upper()]
    for ext in extension_list:
        file_list = glob.glob(path_exptdata+'*'+ext)
        for file_name in tqdm.tqdm(file_list[:], desc='Data Download'):
            # try:
                Intensity_list = read_pattern_in_file(file_name, xaxis_min, xaxis_max, xaxis_step)
                if Intensity_list=='error1':
                    print("="*80)
                    print('Error! 2 theta step of this data in measurement not suppposed. (%s)'%(file_name))
                    print("="*80)
                    continue
                x_test = np.array([Intensity_list], np.float64)
                x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
                x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
                tf.keras.backend.set_floatx('float64')
                x_test_ = x_test[..., tf.newaxis]
                files[file_name] = x_test_
    return files

def result(predicts):
    QC_num = 0
    AC11_num = 0
    AC21_num = 0
    Others_num = 0
    for value in predicts:
        max_value = max(value)
        max_index = np.argmax(value)
        if max_index == 0:
            QC_num += 1
        elif max_index == 1:
            AC11_num += 1
        elif max_index == 2:
            AC21_num += 1
        elif max_index == 3:
            Others_num += 1
    return [QC_num, AC11_num, AC21_num, Others_num]

def preprocess_pattern(Intensity_list):
    x_test = np.array([Intensity_list], np.float64)
    x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
    x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
    tf.keras.backend.set_floatx('float64')
    x_test_ = x_test[..., tf.newaxis]
    return x_test_

def get_TPTNFPFN(QC_prediction_list, AC11_prediction_list, AC21_prediction_list, Others_prediction_list, data_num, label):
    if label=='QC':
        TP = QC_prediction_list[0]
        FP = AC11_prediction_list[0]+AC21_prediction_list[0]+Others_prediction_list[0]
        FN = QC_prediction_list[1]+QC_prediction_list[2]+QC_prediction_list[3]
        TN = data_num-(TP+FP+FN)
    if label=='AC11':
        TP = AC11_prediction_list[1]
        FP = QC_prediction_list[1]+AC21_prediction_list[1]+QC_prediction_list[1]
        FN = AC11_prediction_list[0]+AC11_prediction_list[2]+AC11_prediction_list[3]
        TN = data_num-(TP+FP+FN)
    if label=='AC21':
        TP = AC21_prediction_list[2]
        FP = QC_prediction_list[2]+AC11_prediction_list[2]+Others_prediction_list[2]
        FN = AC21_prediction_list[0]+AC21_prediction_list[1]+AC21_prediction_list[3]
        TN = data_num-(TP+FP+FN)
    if label=='Others':
        TP = Others_prediction_list[3]
        FP = QC_prediction_list[3]+AC11_prediction_list[3]+AC21_prediction_list[3]
        FN = Others_prediction_list[0]+Others_prediction_list[1]+Others_prediction_list[2]
        TN = data_num-(TP+FP+FN)
    return TP, FP, FN, TN

def calc_accuracy(TP_all, data_num):
    accuracy = TP_all/data_num
    return round(accuracy,4)

def calc_recall(TP, FN):
    recall = TP/(TP+FN)
    return recall

def calc_precision(TP, FP):
    precision = TP/(TP+FP)
    return precision

def calc_F1(recall, precision):
    F1_value = (2*recall*precision)/(recall+precision)
    return F1_value

def calc_evaluation(TP, FP, FN, TN):
    accuracy = calc_accuracy(TP, TP+FP+FN+TN)
    recall = calc_recall(TP, FN)
    precision = calc_precision(TP, FP)
    F1_value = calc_F1(recall, precision)
    return round(accuracy,4), round(recall,4), round(precision,4), round(F1_value,4)

def TablePlot(df,outputPath,w,h):
    fig, ax = plt.subplots(figsize=(w,h))
    ax.axis('off')
    table = ax.table(cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc='center',
            bbox=[0,0,1,1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.savefig(outputPath)
    return

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
      tths = np.arange(20,80,0.01)
      plt.ylim(0, 1)
      axs[i][j].plot(tths, intensity_list[a])
      a+=1
  try:
    plt.savefig(outputDir+'/PXRD.png')
  except:
    pass
  return

def generate_prediction_results(data, filename="prediction_results.csv"):
    # Normalize probabilities so they sum up to 1 for each sample
    for i in range(len(data["Sample_Name"])):
        total_prob = data["iQC"][i] + data["1/1AC"][i] + data["2/1AC"][i] + data["Others"][i]
        data["iQC"][i] = round(data["iQC"][i] / total_prob, 4)
        data["1/1AC"][i] = round(data["1/1AC"][i] / total_prob, 4)
        data["2/1AC"][i] = round(data["2/1AC"][i] / total_prob, 4)
        data["Others"][i] = round(data["Others"][i] / total_prob, 4)

    # Organize data as a DataFrame
    df = pd.DataFrame(data)

    # Get the class with the highest probability and the class with the second highest probability
    prob_columns = ["iQC", "1/1AC", "2/1AC", "Others"]
    class_labels = ["iQC", "1/1AC", "2/1AC", "Others"]

    df['Top_Class'] = df[prob_columns].idxmax(axis=1).apply(lambda x: x.replace("Prob_", ""))
    df['Second_Top_Class'] = df[prob_columns].apply(
        lambda row: class_labels[np.argsort(row.values)[-2]], axis=1
    )

    # Create a remark column: mark samples where the second highest probability is >= 0.3
    df['Second_Top_Prob'] = df[prob_columns].apply(lambda row: sorted(row, reverse=True)[1], axis=1)
    df['Remark'] = df['Second_Top_Prob'].apply(lambda x: "High Second Prediction" if x >= 0.3 else "")

    # Display selected columns only
    df = df[[
        "Sample_Name", "Predicted_Label",
        "iQC", "1/1AC", "2/1AC", "Others",
        "Second_Top_Class", "Remark"
    ]]

    print(df)
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")