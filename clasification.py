# 20241104 K. Hiroki

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tqdm
import tensorflow as tf
import sys
from mylib import utility


os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run(path_model, expt_data_path, output_path, extension='.txt', xaxis_min=20.0, xaxis_max=80.0, xaxis_step=0.01, flag_plot=False):
    """
    run model screening
    """

    data_list = utility.data_download(expt_data_path, extension, xaxis_min, xaxis_max, xaxis_step)

    file_path = expt_data_path+'/XRD_plot'
    if os.path.isdir('%s'%(file_path))==False:
        os.mkdir('%s'%(file_path))

    len_path = len(expt_data_path)
    len_extension = len(extension)
    
    screening(path_model, output_path, data_list, expt_data_path, extension)
    
    if flag_plot==True:
        for file in data_list:
            data_name = file[len_path:len(file)-len_extension]
            utility.show_fig(file, xaxis_min, xaxis_max, xaxis_step, data_list[file][0], file_path+'/'+data_name+'.png', 'tth')
    else:
        pass
    return 

def screening(path_model, output_path, patterns, expt_data_path, extension):
    len_path = len(expt_data_path)
    len_extension = len(extension)

    tf.keras.backend.clear_session()
    try:
       model = tf.keras.models.load_model(path_model, compile = False)
    except OSError:
       print("Failed to load model.")
       sys.exit()
    name_max_len = 0
    prediction_results = {
                        "Sample_Name": [],  # Sample Name
                        "Predicted_Label": [],  # Predicted label
                        "iQC": [],  # Predicted value
                        "1/1AC": [],  # Predicted value
                        "2/1AC": [],  # Predicted value
                        "Others": [],  # Predicted value
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
        except:
            print('Screening error:', name)
    utility.generate_prediction_results(prediction_results, output_path)
    return


if __name__ == '__main__':
    ###結果の保存先の変更を忘れずに###    
    path_model = './models/tth/QC_11AC_21AC_Others/single_multi_GAN/iQC1_11AC1_21AC1_Others1_tuninged/TrainData600000/S_hwhm001-01/s4-m5-G1/aico4-6_HWHM30-300/4.0_6.0__15__256'

    expt_data_path = {'unknown': './data/fujino/20240908/'}
    extension = '.txt'
    
    xaxis_min = 20.0
    xaxis_max = 80.0
    xaxis_step = 0.01

    xrd_plot = True
    for label in expt_data_path:
        output_dir = expt_data_path[label]+'/screening_result'
        if os.path.isdir('%s'%(output_dir))==False:
            os.mkdir('%s'%(output_dir))
        output_path = output_dir + '/screening_'+label+'_data_result_Tsai-type_iQC_model.csv'

        print('-'*40,'Screening',label,'data (',expt_data_path[label],')','-'*40)
        run(path_model, expt_data_path[label], output_path, extension, xaxis_min, xaxis_max, xaxis_step, xrd_plot)
