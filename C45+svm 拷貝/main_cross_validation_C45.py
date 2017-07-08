#-*- encoding=utf-8 -*- 
#交叉驗證_五折 


import numpy as np
import os
import scipy 
import SVM
import csv
import time




def read_csv(filename):
    #讀入csv
    inFile = filename
    list = []
    csvfile = file(inFile, 'rb')
    reader = csv.reader(csvfile)
    for line in reader:
        list.append(line)
    #inFile = scipy.loadtxt(inFile, delimiter = ",")
    #轉為array
    inFile = np.array(list)
    print np.shape(inFile)
    return inFile

#找dict 值大於頻均
def find_key(input_dict, f):
    match_data = {}
    for (key, value) in input_dict.items():
        if value >= f:
            match_data[key] = value
    return (match_data.keys())


   
def cross_validation(file):
    n = int(0.2 * datanp.shape[0]) #抽樣以五折驗證，所以乘0.2
    mean_accuracy_svm = 0
    fs_list = []
    fs_dict = {} #計算每一折的list 總list feature list 出現結果

    for i in range(5):
        i=i+1
        print str(i)+"-fold"
        if(i == 1): #第一折
            idx = range(0,n)     
        elif(i == 2): #第二折
            idx = range(n,n*2)
        elif(i == 3):#第三折
            idx = range(n*2,n*3)
        elif(i == 4):#第四折
            idx = range(n*3,n*4)
        else:#第五折
            idx = range(n*4,file.shape[0])
        testdata = file[idx,:]
        idx_IN_columns = [i for i in xrange(np.shape(file)[0]) if i not in idx]
        traindata=file[idx_IN_columns,:]
    
        
        #accuracy_svm,feature_select_index_list = SVM.SVM_main(traindata,testdata)
        feature_select_index_list = SVM.SVM_main(traindata,testdata)
        #mean_accuracy_svm = accuracy_svm + mean_accuracy_svm
        fs_list.extend(feature_select_index_list)
        
    #計算總結果list出現頻率
    for i in fs_list:
        if fs_list.count(i)>1:
            fs_dict[i] = fs_list.count(i)
    print 'feature frequency:'+ str(fs_dict) #fs_dict 顯示feature 各自出現的頻率
    f=sum(list(fs_dict.values()))/len(list(fs_dict.values())) #計算出現頻率的平均 f 門檻值
    print 'feature frequency avg:' + str(f)
    
    fs_list = find_key(fs_dict, f)
    print  'feature result:'+ str(fs_list)
    #mean_accuracy_result_svm =mean_accuracy_svm / 5 
    #print "SVM 5-fold mean result accuracy: %s" %  mean_accuracy_result_svm
    

    



if __name__=='__main__':
    filename = "iris" + ".csv"
    tStart = time.time()#計時開始
    datanp=read_csv("/Users/chencherry/Desktop/lab_code/b_data/" + filename)
    print "Filename: " + filename
    cross_validation(datanp)
    tEnd = time.time()#計時結束
    print "It cost %f sec" % (tEnd - tStart)#會自動做近位