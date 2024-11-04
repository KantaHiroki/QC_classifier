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


def read_pattern_in_file(path_exptdata, file_path):
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
        if 20 <= tth < 80:
            tth_list.append(tth)
            Intensity = float(line_list[1])
            Intensity_list.append(Intensity)
    f.close()

    # print(file_path, len(Intensity_list))
    if len(Intensity_list)<6000:
        tth_interval = tth_list[1]-tth_list[0]
        Intensity_list = data_compensate(tth_list, Intensity_list, tth_interval)
    # if len(Intensity_list)!=6000:
    #     return 'error1'
    return Intensity_list


def data_download_from_list(path_exptdir, path_file_list):
    files = {}
    count_error = 0
    file_list = open(path_file_list, 'r', encoding="utf-8")
    # print(path_exptdir+file_list.readlines()[1][:-1])
    # len_file_list = len(file_list.readlines())
    for file_name in file_list.readlines():
        try:
            tth_list = []
            Intensity_list = []
            # print(path_exptdir+file_name[:-1])
            f = open(path_exptdir+file_name[:-1], 'r', encoding="shift-jis")
            #f = open(file_name, 'r')
            #print(f.readlines()[300])
            for line in f.readlines():
                # try:
                    if line[0] == '*':
                        continue
                    elif line[0]=='#':
                        continue
                    line_list = line[:-1].split()
                    tth = float(line_list[0])
                    if 20 <= tth < 80:
                        tth_list.append(tth)
                        Intensity = float(line_list[1])
                        Intensity_list.append(Intensity)
                #except:
                #    pass
            f.close()

            if len(Intensity_list)!=6000:
                print("=============================")
                print('Error! Please check the file. (%s)'%(file_name))
                print("=============================")
                # file_list.remove(file_name)
                # count_error += 1
                # print(count_error, '/', len_file_list, ', error / file')
                # tth_interval = tth_list[1]-tth_list[0]
                # Intensity_list = data_compensate(tth_list, Intensity_list, tth_interval)

            x_test = np.array([Intensity_list], np.float64)
            x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
            x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
            tf.keras.backend.set_floatx('float64')
            x_test_ = x_test[..., tf.newaxis]
            files[file_name] = x_test_
        except:
            print("=============================")
            print('Error! Please check the file.')
            print(file_name)
            print("=============================")
            # file_list.remove(file_name)
            # count_error += 1
            # print(count_error, '/', len_file_list, ', error / file')
    file_list.close()
    return files

def data_download(path_exptdata, extension):
    files = {}
    extension_list = [extension, extension.upper()]
    for ext in extension_list:
        file_list = glob.glob(path_exptdata+'*'+ext)
        len_file_list = len(file_list)
        count_error = 0
        for file_name in tqdm.tqdm(file_list[:], desc='Data Download'):
            # try:
                Intensity_list = read_pattern_in_file(path_exptdata, file_name)
                if Intensity_list=='error1':
                    print("="*80)
                    print('Error! Please check data length, expect 6000 (%s)'%(file_name))
                    print("="*80)
                    continue
                x_test = np.array([Intensity_list], np.float64)
                x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
                x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
                tf.keras.backend.set_floatx('float64')
                x_test_ = x_test[..., tf.newaxis]
                files[file_name] = x_test_
            # except:
            #     print("=============================")
            #     print('Error! Please check the file.')
            #     print(file_name[len(path_exptdata):])
            #     print("=============================")
            #     file_list.remove(file_name)
            #     count_error += 1
            #     print(count_error, '/', len_file_list, ', error / file')
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