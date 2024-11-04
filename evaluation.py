from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from curses import tparm
from fnmatch import fnmatch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
import math
import tqdm
import generator as gen
import tensorflow as tf


dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059


# def result_4class(predicts):
#     num1 = 0
#     num2 = 0
#     num4 = 0
#     num3 = 0
#     for values in predicts:
#         max_index = np.argmax(values)
#         if max_index == 0:
#             num1 += 1
#         elif max_index == 1:
#             num2 += 1
#         elif max_index == 2:
#             num3 += 1
#         elif max_index == 3:
#             num4 += 1
#     return [num1, num2, num3, num4]

def result_4class(predicts):
    num1 = 0
    num2 = 0
    num4 = 0
    num3 = 0
    for values in predicts:
        max_index = np.argmax(values)
        if max_index == 1:
            if values[max_index]>=0.7:
                pass
            else:
                max_index = values.argsort()[::-1][1]
        if max_index == 0:
            num1 += 1
        elif max_index == 1:
            num2 += 1
        elif max_index == 2:
            num3 += 1
        elif max_index == 3:
            num4 += 1
    return [num1, num2, num3, num4]

def confusion_matrix_2class(QC, Others, outputpath, labels):
    cm=np.array([[QC[0], QC[1]], [Others[0], Others[1]]])
    # plot the results
    fig, (ax1) = plt.subplots(1,  figsize=(13, 5), dpi=150)
    cmd = ConfusionMatrixDisplay(cm,display_labels=labels)
    cmd.plot(
        cmap=plt.cm.Blues,
        ax=ax1
        )
    plt.show()
    fig.savefig(outputpath)
    return

def confusion_matrix_4class(QC, AC11, AC21, Others, outputpath, labels):
    cm=np.array([[QC[0], QC[1], QC[2], QC[3]], [AC11[0], AC11[1], AC11[2], AC11[3]], [AC21[0], AC21[1], AC21[2], AC21[3]], [Others[0], Others[1], Others[2], Others[3]]])
    # plot the results
    fig, (ax1) = plt.subplots(1,  figsize=(13, 5), dpi=150)
    cmd = ConfusionMatrixDisplay(cm,display_labels=labels)
    cmd.plot(
        cmap=plt.cm.Blues,
        ax=ax1
        )
    plt.show()
    fig.savefig(outputpath)
    return

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

def test_datasets_2class(path_test_data, data_num_qc, data_num_nonqc):
    # Multi-iQC dataset
    QC_test = np.load(path_test_data[0])

    # Non-iQC dataset
    others_test = np.load(path_test_data[1])

    return QC_test, others_test

def main(path_model, dic_test_data, data_num_QC, data_num_nonQC, tth_step_num, labels):
    """
    run model evaluation using synthetic datasets
    """
    for name in dic_test_data:
        # Generating test data
        MultiQC_test, others_test = test_datasets_2class(dic_test_data[name], data_num_QC, data_num_nonQC)
        MultiQC_test = MultiQC_test.reshape(data_num_QC, tth_step_num, 1)
        others_test = others_test.reshape(data_num_nonQC, tth_step_num, 1)

        # test trained model
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(path_model, compile = False)

        QC_prediction_list = result(model.predict(MultiQC_test))
        Others_prediction_list = result(model.predict(others_test))

        file_path = path_model+'/evaluation_'+name
        if os.path.isdir('%s'%(file_path))==False:
            os.mkdir('%s'%(file_path))

        confusion_matrix_2class(QC_prediction_list, Others_prediction_list, file_path+'/ConfusionMatrix.png', labels)
        # confusion_matrix_4class(QC_prediction_list, AC11_prediction_list, AC21_prediction_list, Others_prediction_list, file_path+'/ConfusionMatrix.png', labels)

    # print('    TP  FN  FP  TN', file = f)
    # TP_sum = 0
    # FN_sum = 0
    # FP_sum = 0
    # TN_sum = 0
    # for i in Result_QC.keys():
    #     TP = np.count_nonzero(Result_QC[i])
    #     FN = data_num_QC - TP
    #     FP = np.count_nonzero(Result_others[i])
    #     TN = data_num_nonQC - FP
    #     print('{:<.3f}'.format(i), '{:<5d}'.format(TP), '{:<5d}'.format(FN),'{:<5d}'.format(FP),'{:<5d}'.format(TN), file = f)
    #     TP_sum += TP
    #     FN_sum += FN
    #     FP_sum += FP
    #     TN_sum += TN
    # print('\n', file = f)
    # print('ALL   TP  FN  FP  TN', file = f)
    # print('{:<5d}'.format(TP_sum), '{:<5d}'.format(FN_sum),'{:<5d}'.format(FP_sum),'{:<5d}'.format(TN_sum), file = f)
    # f.close()
    return
    
if __name__ == '__main__':
    
    path_model = './models/decagonal_QC/a36-48_c38-56/GAN30/aico4-6_HWHM30-300/4.0_6.0__30__256'
    
    tth_step_num = 6000

    data_num_QC = 500
    data_num_nonQC = 500

    labels = ['dQC', 'Others']

    dic_test_data = {'single': ['./data/decagonal/a36-48_c38-56/single_data500.npy',
                                './data/decagonal/Others/others_500.npy'],
                    'multi': ['./data/decagonal/a36-48_c38-56/multi_data500.npy',
                                './data/decagonal/Others/others_500.npy'],
                    'GAN': ['./data/decagonal/a36-48_c38-56/GAN_data500.npy',
                                './data/decagonal/Others/others_GAN_500.npy']}

    main(path_model, dic_test_data, data_num_QC, data_num_nonQC, tth_step_num, labels)