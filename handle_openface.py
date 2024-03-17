import json
import os
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import linregress
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import pandas as pd
from mmsdk import mmdatasdk
import seaborn as sns
import random
def get_lstsq(x, y):
    #plt.plot(x,y,'x')
    X = np.hstack((x[:, np.newaxis], np.ones((x.shape[-1], 1))))
    C, resid, rank, s = lstsq(X, y)
    #print(C, resid, rank, s)
    #p = plt.plot(x, y, 'rx')
    #p = plt.plot(x, C[0]*x+C[1], 'k--')
    # print(x)
    # print(C)
    return C

def json_to_dict(json_file, dict, max_num):

    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if content['people'] == []:
                pass
            else:
                people_inf = content['people']

                if json_file.__contains__("_000000000000_keypoints.json"):
                    dict['max_conf_num'] = max_num
                    try:
                        inf_keys = people_inf[max_num].keys()
                        for k in inf_keys:
                            if k != 'person_id':
                                dict[k] = people_inf[max_num][k]
                    except IndexError:
                        pass
                else:
                    for k in dict.keys():
                        if k not in ('video_name', 'max_conf_num'):
                            tmp = dict[k]
                            try:
                                tmp = np.hstack((tmp, people_inf[max_num][k]))
                                # print("---------before11-------------")
                                # print(len(dict[k]))
                                dict[k] = tmp.tolist()
                                # print("---------after11-------------")
                                # print(len(dict[k]))
                            except IndexError:
                                pass
                    # print("---------------------------------")
                    # print(dict)

    else:
        pass
    return dict
    # print("Person {} has the maximum confidence!!".format(max_conf_num)) #num start at 0


def write_move2json(file_name, v_movements):
    if not os.path.exists(file_name):
        with open(file_name, 'w', encoding="utf-8") as f:
            print("The output json file does not exists!")
    with open(file_name, "r", encoding="utf-8") as f:
        file = f.read()
        if len(file) > 0:
            old_data = json.loads(file)
            old_data = str(old_data) + "," + str(v_movements)
        else:
            old_data = str(v_movements)

    with open(file_name, 'w', encoding="utf-8") as f:
        json.dump(old_data, f)

def write_AUs2csv(data_type):
    dir = "E:\MER\MER2023-Baseline-master\dataset-process/features"
    dirname = "openface_all_" + data_type
    feature_dir = os.path.join(dir, dirname)
    output_filename = data_type + "_meld_au.csv"
    output_file = os.path.join(dir, output_filename)
    files = os.listdir(feature_dir)
    rowss = []
    for f in files:
        file_name = os.path.join(feature_dir, f, f+".csv")
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df = df.filter(regex='AU')
            df_t = df.apply(lambda x:x.mean())
            df_t['v_name'] = f
            tmp_row = []
            for row in df_t:
                tmp_row.append(row)

            rowss.append(tmp_row)

    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rowss)

    print("ALL done!!!")

def write_text_for_AUs(data_type):
    AU_word_mapping = {'AU01_c': "Inner Brow Raiser",
                       'AU02_c': "Outer Brow Raiser",
                       'AU04_c': "Brow Lowerer",
                       'AU05_c': "Upper Lid Raiser",
                       'AU06_c': "Cheek Raiser",
                       'AU07_c': "Lid Tightener",
                       'AU09_c': "Nose Wrinkler",
                       'AU10_c': "Upper Lip Raiser",
                       'AU12_c': "Nasolabial Deepener",
                       'AU14_c': "Dimpler",
                       'AU15_c': "Lip Corner Depressor",
                       'AU17_c': "Chin Raiser",
                       'AU20_c': "Lip stretcher",
                       'AU23_c': "Lip Tightener",
                       'AU25_c': "Lips part",
                       'AU26_c': "Jaw Drop",
                       'AU28_c': "Lip Suck",
                       'AU45_c': "Blink"}
    dir = "E:\MER\MER2023-Baseline-master\dataset-process/features"
    input_filename = data_type + "_meld_au.csv"
    input_file = os.path.join(dir, input_filename)
    df = pd.read_csv(input_file)
    df_au = df.loc[:, ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c',
       'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
       'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c',
       'sentiment', 'v_name']]
    output_filename = data_type + "_meld_au_text.csv"
    v_text_csv = os.path.join(dir, output_filename)
    with open(v_text_csv, 'w') as file:
        file.write("name,Sentiment,Utterance")
        file.write("\n")
        for index, row in df_au.iterrows():
            tmp_v = []
            tmp_v.append(row['v_name'])
            tmp_v.append(row['sentiment'])
            aus = ""
            for k in AU_word_mapping.keys():
                if row[k] > 0:
                    aus += AU_word_mapping[k] + "+"
            if aus.replace("+","") != "":
                tmp_v.append(aus.lower())
                tmp_v = str(tmp_v).strip().replace("[","").replace("]","").replace("'","")[:-1]
                file.write(tmp_v)
                file.write("\n")
    print("ALL DONE!!")

def write_text_for_AUs_intensity(data_type):
    AU_word_mapping = {'AU01_c': "Inner Brow Raiser",
                       'AU02_c': "Outer Brow Raiser",
                       'AU04_c': "Brow Lowerer",
                       'AU05_c': "Upper Lid Raiser",
                       'AU06_c': "Cheek Raiser",
                       'AU07_c': "Lid Tightener",
                       'AU09_c': "Nose Wrinkler",
                       'AU10_c': "Upper Lip Raiser",
                       'AU12_c': "Nasolabial Deepener",
                       'AU14_c': "Dimpler",
                       'AU15_c': "Lip Corner Depressor",
                       'AU17_c': "Chin Raiser",
                       'AU20_c': "Lip stretcher",
                       'AU23_c': "Lip Tightener",
                       'AU25_c': "Lips part",
                       'AU26_c': "Jaw Drop",
                       'AU28_c': "Lip Suck",
                       'AU45_c': "Blink"}
    AU_intensity_rows = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r'
        ,'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r'
        , 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    s_words = ["slightly"
                          ,"somewhat"
                          ,"a little"
                          ,"minially"]
    m_words = ["moderately"
                          ,"fairly"
                          ,"reasonably"
                          ,"quite"
                          ,"in part"]
    l_words = ["extremely"
                          ,"intensely"
                          ,"passionately"
                          ,"overwhelmingly"
                          ,"exceedingly"
                          ,"profoundly"
                          ,"fiercely"]
    dir = "E:\MER\MER2023-Baseline-master\dataset-process/features"
    input_filename = data_type + "_meld_au.csv"
    input_file = os.path.join(dir, input_filename)
    df = pd.read_csv(input_file)
    # df_au = df.loc[:, ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c',
    #    'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
    #    'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c',
    #    'sentiment', 'v_name']]
    output_filename = data_type + "_meld_au_text_i.csv"
    v_text_csv = os.path.join(dir, output_filename)
    with open(v_text_csv, 'w') as file:
        file.write("name,Sentiment,Utterance_i")
        file.write("\n")
        for index, row in df.iterrows():
            tmp_v = []
            tmp_v.append(row['v_name'])
            tmp_v.append(row['sentiment'])
            aus = ""
            for k in AU_word_mapping.keys():
                if row[k] > 0:
                    aus += AU_word_mapping[k]
                    k_row = k.split("_")[0] + "_r"
                    if k_row in df.columns:
                        if row[k_row] > 0 and row[k_row] <= 2:
                            num = random.randint(0,3)
                            AU_i_word = s_words[num]
                        elif row[k_row] > 2 and row[k_row] <= 3:
                            num = random.randint(0,4)
                            AU_i_word = m_words[num]
                        elif row[k_row] > 3 and row[k_row] <= 5:
                            num = random.randint(0, 6)
                            AU_i_word = l_words[num]

                        aus += " " + AU_i_word

                    aus += "+"


            if aus.replace("+","") != "":
                tmp_v.append(aus.lower())
                tmp_v = str(tmp_v).strip().replace("[","").replace("]","").replace("'","")[:-1]
                file.write(tmp_v)
                file.write("\n")
    print("ALL DONE!!")

if __name__ == '__main__':



    #Writing the AUs and corresponding semantic description for AUs to csv file.
    # write_AUs2csv("test")
    #write_text_for_AUs("train")
    #write_text_for_AUs("dev")
    #write_text_for_AUs("test")

    #write_text_for_AUs_intensity("dev")
    write_text_for_AUs_intensity("train")
    write_text_for_AUs_intensity("test")



    # df_au_r = df.loc[:, ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r',
    #                      'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r',
    #                      'AU45_r','sentiment', 'v_name']]
    # group_result = df_au_r.groupby(['sentiment'])
    # tmp1 = group_result.median()
    # print(tmp1)







    # for i in ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c',
    #    'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
    #    'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']:
    #     tmp = group_result[i].mean()
    #     print(tmp)
    #print(df_brow[df_au['total'] > 0].count())


    # points = pd.read_csv(meld_b_csv)
    # df = pd.DataFrame(points)
    # print(df.size)
    # sns.scatterplot(x='brow-total', y='eye-total', data=df, hue='sentiment', style='sentiment', s=5)






