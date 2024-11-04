#@title Screening for experimental data

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import glob
import tqdm
#import generator as gen
import tensorflow as tf
import matplotlib.pyplot as plt
from curses import tparm
from fnmatch import fnmatch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import codecs
from mylib import save_fig

os.environ["CUDA_VISIBLE_DEVICES"]="1"

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059










def run(path_model, expt_data_path, extension, output_path, xaxis_min, xaxis_max, xaxis_step, flag_plot):
    """
    run model evaluation using synthetic datasets 
    """

    data_list = data_download(expt_data_path, extension)
    # data_list = data_download_from_list('./data/Tamura_lab/expt_data/QC/', './data/Tamura_lab/expt_data/indexing/indexed_QC_list.txt')

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(path_model, compile = False)
    #try:
    #    model = keras.models.load_model("./models/junk/test_model.hdf5")
    #except OSError:
    #    print("Failed to load model.")
    #    sys.exit()

    file_path = expt_data_path+'/XRD_plot'
    if os.path.isdir('%s'%(file_path))==False:
        os.mkdir('%s'%(file_path))

    len_path = len(expt_data_path)
    len_extension = len(extension)
    
    prediction_num = screening(path_model, output_path, data_list, xaxis_min, xaxis_max, xaxis_step, expt_data_path, extension)
    
    # for data in data_list:
    #   #print(model.predict(data_list[data]))
    #   #prediction = result(model.predict(data_list[data]))
    #   print('--------------------------------------------------------------------------------------------------')
    #   print(data)
    #   #print('Result: [iQC, 1/1AC, 2/1AC, Others] ==> ', prediction)
    #   print('Result: [iQC, 1/1AC, 2/1AC, Others] ==> ', model.predict(data_list[data]))
    #   print('--------------------------------------------------------------------------------------------------')
    #   data_name = data[len_path:len(data)-len_extension]
    #   show_fig(data_name, xaxis_min, xaxis_max, xaxis_step, data_list[data][0], file_path+'/'+data_name+'.png', flag)

    
    # a, b = 0, 0
    if flag_plot==True:
        for file in data_list:
            data_name = file[len_path:len(file)-len_extension]
            save_fig.show_fig(file, xaxis_min, xaxis_max, xaxis_step, data_list[file][0], file_path+'/'+data_name+'.png', 'tth')
    else:
        pass
    
    return prediction_num

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

def screening(model_path, output_path, patterns, xaxis_min, xaxis_max, xaxis_step, expt_data_path, extension):
    QC_list = []
    AC11_list = []
    AC21_list = []
    others_list = []
    xaxis_array = np.arange(xaxis_min, xaxis_max, xaxis_step)
    xaxis_step_num = int((xaxis_max - xaxis_min)/xaxis_step)
    data_num = len(patterns)
    len_path = len(expt_data_path)
    len_extension = len(extension)

    model = tf.keras.models.load_model(model_path, compile = False)
    name_max_len = 0
    a = 0
    prediction_results = {
                        "Sample_Name": [],  # Sample Name
                        "Predicted_Label": [],  # Predicted label
                        "iQC": [],
                        "1/1AC": [],
                        "2/1AC": [],
                        "Others": [],
                        }
    for name in tqdm.tqdm(patterns, desc='Screening'):
        try:
            # if a==data_num-32:
            #     break
            prediction_value = model.predict(patterns[name], verbose=0)
            prediction = np.argmax(prediction_value)
            data_name = name[len_path:len(name)-len_extension]
            if name_max_len < len(data_name):
                name_max_len = len(data_name)
            prediction_results['Sample_Name'].append(data_name)
            if prediction==0:
                prediction_results['Predicted_Label'].append('iQC')
            elif prediction==1:
                prediction_results['Predicted_Label'].append('1/1AC')
            elif prediction==2:
                prediction_results['Predicted_Label'].append('2/1AC')
            elif prediction==3:
                prediction_results['Predicted_Label'].append('Others')
            prediction_results['iQC'].append(prediction_value[0][0])
            prediction_results['1/1AC'].append(prediction_value[0][1])
            prediction_results['2/1AC'].append(prediction_value[0][2])
            prediction_results['Others'].append(prediction_value[0][3])
            a+=1
        except:
            print('Screening error:', name)
    generate_prediction_results(prediction_results, output_path)
    return


# def screening(model_path, output_path, patterns, xaxis_min, xaxis_max, xaxis_step, expt_data_path, extension):
#     QC_list = []
#     AC11_list = []
#     AC21_list = []
#     others_list = []
#     xaxis_array = np.arange(xaxis_min, xaxis_max, xaxis_step)
#     xaxis_step_num = int((xaxis_max - xaxis_min)/xaxis_step)
#     data_num = len(patterns)
#     len_path = len(expt_data_path)
#     len_extension = len(extension)

#     model = tf.keras.models.load_model(model_path, compile = False)
#     name_max_len = 0
#     a = 0
#     for name in tqdm.tqdm(patterns, desc='Screening'):
#         try:
#             # if a==data_num-32:
#             #     break
#             prediction_value = model.predict(patterns[name], verbose=0)
#             prediction = np.argmax(prediction_value)
#             data_name = name[len_path:len(name)-len_extension]
#             if name_max_len < len(data_name):
#                 name_max_len = len(data_name)
#             if prediction==0:
#                 QC_list.append([data_name, prediction_value[0]])
#             elif prediction==1:
#                 AC11_list.append([data_name, prediction_value[0]])
#             elif prediction==2:
#                 AC21_list.append([data_name, prediction_value[0]])
#             elif prediction==3:
#                 others_list.append([data_name, prediction_value[0]])
#             a+=1
#         except:
#             print('Screening error:', name)

#     f = open(output_path, 'w', encoding="utf-8", errors="ignore")
#     blank = name_max_len + 5
#     print('Screening Result', file=f)
#     print('  Data Num.: ', len(patterns), file=f)
#     print('='*(blank+70), file = f)
#     if len(QC_list)>0:
#       print('  Detection iQC  (', str(len(QC_list)), 'data)', file = f)
#       print('    File name', ' '*(name_max_len+7), 'Prediction value', file = f)
#       print(' '*(15+name_max_len), ' iQC    1/1AC    2/1AC    Others', file = f)
#       print('-'*(blank+70), file = f)
#       for info in sorted(QC_list, reverse=True, key=lambda x: x[1][0]):
#         em = blank-len(info[0])+3
#         print('     ', info[0], ' '*em, format(info[1][0],'.5f'), format(info[1][1],'.5f'), format(info[1][2],'.5f'), format(info[1][3],'.5f'), file = f)
#       print('='*(blank+70), file = f)
#     if len(AC11_list)>0:
#       print('  Detection 1/1AC  (', str(len(AC11_list)), 'data)', file = f)
#       print('    File name', ' '*(name_max_len+7), 'Prediction value', file = f)
#       print(' '*(15+name_max_len), ' iQC    1/1AC    2/1AC    Others', file = f)
#       print('-'*(blank+70), file = f)
#       for info in sorted(AC11_list, reverse=True, key=lambda x: x[1][1]):
#         em = blank-len(info[0])+3
#         print('     ', info[0], ' '*em, format(info[1][0],'.5f'), format(info[1][1],'.5f'), format(info[1][2],'.5f'), format(info[1][3],'.5f'), file = f)
#       print('='*(blank+70), file = f)
#     if len(AC21_list)>0:
#       print('  Detection 2/1AC  (', str(len(AC21_list)), 'data)', file = f)
#       print('    File name', ' '*(name_max_len+7), 'Prediction value', file = f)
#       print(' '*(15+name_max_len), ' iQC    1/1AC    2/1AC    Others', file = f)
#       print('-'*(blank+70), file = f)
#       for info in sorted(AC21_list, reverse=True, key=lambda x: x[1][2]):
#         em = blank-len(info[0])+3
#         print('     ', info[0], ' '*em, format(info[1][0],'.5f'), format(info[1][1],'.5f'), format(info[1][2],'.5f'), format(info[1][3],'.5f'), file = f)
#       print('='*(blank+70), file = f)
#     if len(others_list)>0:
#       print('  Detection Others  (', str(len(others_list)), 'data)', file = f)
#       print('    File name', ' '*(name_max_len+7), 'Prediction value', file = f)
#       print(' '*(15+name_max_len), ' iQC    1/1AC    2/1AC    Others', file = f)
#       print('-'*(blank+70), file = f)
#       for info in sorted(others_list, reverse=True, key=lambda x: x[1][3]):
#         em = blank-len(info[0])+3
#         print('     ', info[0], ' '*em, format(info[1][0],'.5f'), format(info[1][1],'.5f'), format(info[1][2],'.5f'), format(info[1][3],'.5f'), file = f)
#       print('='*(blank+70), file = f)
#     f.close()
#     return [len(QC_list), len(AC11_list), len(AC21_list), len(others_list)]

    
if __name__ == '__main__':

    ###結果の保存先の変更を忘れずに###

    flag = 'tth'
    
    # path_model = './models/tth/QC_11AC_21AC_Others/expt_25%/aico4-6_HWHM30-300/4.0_6.0__20__256'
    path_model = './models/tth/QC_11AC_21AC_Others/single_multi_GAN/iQC1_11AC1_21AC1_Others1_tuninged/TrainData600000/S_hwhm001-01/s4-m5-G1/aico4-6_HWHM30-300/4.0_6.0__15__256'
    # path_model = './models/tth/QC_11AC_21AC_Others/expt_25%/data_optimized20240602/aico4-6_HWHM30-300/4.0_6.0__20__256'

    # expt_data_path = {'unknown': './data/fujino/20240908/'}
    
    expt_data_path = {'iQC': './data/experimental_evaluation_dataset/iQC/',
                    '11AC': './data/experimental_evaluation_dataset/AC11/',
                    '21AC': './data/experimental_evaluation_dataset/AC21/',
                    'Others': './data/experimental_evaluation_dataset/Others/'
                    }

    extension = '.txt'

    # output_dir = expt_data_path+'/screening_result'
    
    # output_path = output_dir + '/screening_iQC_result_Multimodel.txt'
    # output_path = output_dir + '/screening_iQC_result_multimodel.txt'

    aico = 5.0
    aico_min = 4.0
    aico_max = 6.0
    aico_delta = 2.0
    tth_min = 20.0
    tth_max = 80.0
    hklmno_range = 6
    wvl = dic_wvl['Cu_Ka']
    qperp_cutoff = 1.5  # in r.l.u (Yamamoto's setting).  this corresponds to 1.5*sqrt(2)=2.12... in r.l.u in Cahn-Gratias setting. 
    data_num_QC    = 1000 # number of QC patterns for each single model
    data_num_nonQC = 1000 # number of non-QCpatterns for each single model
    
    xaxis_min, xaxis_max, xaxis_step, hwhm_min, hwhm_max = parameters(flag, aico_min, aico_max, tth_min, tth_max)

    xrd_plot = True
    prediction_num = []
    for label in expt_data_path:
        # output_dir = expt_data_path[label]+'/screening_result'
        output_dir = path_model+'/evaluation_extp'
        if os.path.isdir('%s'%(output_dir))==False:
            os.mkdir('%s'%(output_dir))
        output_path = output_dir + '/screening_'+label+'_data_result_S_hwhm001-01_s4-m5-G1model.csv'

        print('-'*40,'Screening',label,'-'*40)
        prediction_num.append(run(path_model, expt_data_path[label], extension, output_path, xaxis_min, xaxis_max, xaxis_step, xrd_plot))
    # evaluation.confusion_matrix_4class(prediction_num[0], prediction_num[1], prediction_num[2], prediction_num[3], output_dir+'/ConfusionMatrix.png', list(expt_data_path.keys()))
    # accuracy = calc_accuracy(prediction_num[0][0]+prediction_num[1][1]+prediction_num[2][2]+prediction_num[3][3], sum(prediction_num[0])+sum(prediction_num[1])+sum(prediction_num[2])+sum(prediction_num[3]))
    # print('Accuracy : ', accuracy)