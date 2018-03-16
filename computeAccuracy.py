# -*- coding: utf-8 -*-  

  

import sys
import string
import commands
import re
import os
import copy
import math
import csv
import codecs
import util




def getMatchPeriod(pattern,text):
    match = re.findall(pattern,text)
    if len(match)>0:
        return match[0]
    else:
        #print text+" match error"
        return []

def saveToCSV(fileName,dataArray):
    csvfile = file(fileName, 'wb')
    csvfile.write(codecs.BOM_UTF8)
    writer = csv.writer(csvfile)
    writer.writerows(dataArray)
    csvfile.close()


#读取主观评测结果文件
def readResultCSV():
    path = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/compute_accuracy/mos_file/split_mos_new.csv'
    csvfile = file(path, 'rb')
    reader = csv.reader(csvfile)

    rstDic= {}


    for lines in reader:
        url = lines[0]
        rst = lines[1]
        name = os.path.basename(url)
        rstDic[name] = rst
        # if rst == '2':
        #     #util.log(name)
        #     frompath = os.path.join(ORI_PATH,name)
        #     topath = TO_PATH
        #     util.run_command('cp %s %s'%(frompath,topath))
    csvfile.close()
    return rstDic


def readPredictResult():
    path = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/compute_accuracy/predict_result/al_result.csv'
    csvfile = file(path, 'rb')
    reader = csv.reader(csvfile)

    rstDic= {}


    for lines in reader:
        name = lines[0]
        rst = lines[3]
        rstDic[name] = rst
    csvfile.close()
    return rstDic


ORI_PATH = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/testaccuracy'
TO_PATH = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/compute_accuracy/test_split_predict/al'


def main():
    rstDic = readResultCSV()
    predictDic = readPredictResult()
    ac_num = 0    #正确检测分屏数量
    split_num = 0  #分屏的数量
    lost_num= 0    #漏掉检测分屏的数量
    error_num = 0  #错误检测为分屏的数量
    total_num = len(predictDic.keys())
    no_split_num = 0 #没分屏数量
    no_split_right_num = 0 #没分屏预测正确

    path_1_1 = os.path.join(TO_PATH,'1_1')
    path_1_0 = os.path.join(TO_PATH, '1_0')
    path_0_1 = os.path.join(TO_PATH, '0_1')
    path_0_0 = os.path.join(TO_PATH, '0_0')
    util.make_dir(path_1_1)
    util.make_dir(path_1_0)
    util.make_dir(path_0_1)
    util.make_dir(path_0_0)


    for key in predictDic.keys():
        frompath = os.path.join(ORI_PATH, key)
        if int(rstDic[key]) == 2:
            split_num +=1
            if int(predictDic[key]) == 1:
                ac_num += 1
                util.run_command('cp %s %s' % (frompath, path_1_1))
            else:
                lost_num +=1
                util.run_command('cp %s %s' % (frompath, path_1_0))

        else:
            no_split_num +=1
            if int(predictDic[key]) == 1:
                error_num +=1
                util.run_command('cp %s %s' % (frompath, path_0_1))
            else:
                no_split_right_num +=1
                #util.run_command('cp %s %s' % (frompath, path_0_0))






    util.log('tatal:%d accuracy: %d  lost: %d  error: %d right_nosplit:%d'%(total_num,ac_num,lost_num,error_num,no_split_right_num))







    
def update_mos_file():
    mos_dict = readResultCSV()
    path = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/compute_accuracy/test_split_predict/split3/0_1'
    for name in os.listdir(path):
        if mos_dict.has_key(name):
            util.log('update mos name:%s  value:%s'%(name,mos_dict[name]))
            mos_dict[name] = '2'
        else:
            util.log('no file :%s'%name)

    csvfile = file('split_mos_new.csv', 'wb')
    writer = csv.writer(csvfile)
    for key in mos_dict.keys():
        writer.writerow([key,mos_dict[key]])
    csvfile.close()




    


if __name__ == '__main__':
    main()
    #update_mos_file()
















