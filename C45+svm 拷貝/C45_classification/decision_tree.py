#-*- coding:utf-8 -*-
import sys
from math import log
import operator
from numpy import mean
import numpy as np


def get_labels(train_file):
    '''
    返回所有數據集class labels(列表)
    '''
    n=train_file.shape[1]
    n = n - 1
    labels= train_file[:,n]
    labels=labels.tolist()
    #print labels
    return labels

def format_data(dataset_file):
    '''
    返回dataset(列表集合)和features(列表)
    '''
    n=dataset_file.shape[1]
    dataset =dataset_file[:,:n]
    dataset=dataset.tolist()
    #轉換型態為float data & str class_label
    list = np.array(dataset)
    n =(list.shape[1]) - 1
    t_num = list.shape[0]
    data = np.array(list[:,0:n],float)
    data= data.tolist()
    class_label =  np.array(list[:,n],str)
    class_label = class_label.tolist()
    for row in range(t_num):
        data[row].append(class_label[row])
    dataset= data
    fs_num=dataset_file.shape[1] - 1
    features_list = []
    for i in range(fs_num):
        features_list.append("a"+str(i+1)) 
    features =features_list #feature 屬性列表
    # print features
    return dataset,features,fs_num

def split_dataset(dataset,feature_index,labels,fs_num):
    '''
    按指定feature劃分數據集，返回四個列表:
    @dataset_less:指定特徵項的屬性值＜=該特徵項平均值的子數據集
    @dataset_greater:指定特徵項的屬性值＞該特徵項平均值的子數據集
    @label_less:按指定特徵項的屬性值＜=該特徵項平均值切割後子標籤集
    @label_greater:按指定特徵項的屬性值＞該特徵項平均值切割後子標籤集
    '''
    dataset_less = []
    dataset_greater = []
    label_less = []
    label_greater = []
    datasets = []
    for data in dataset:
        datasets.append(data[0:fs_num])
    mean_value = mean(datasets,axis = 0)[feature_index]   #數據集在該特徵項的所有取值的平均值
    for data in dataset:
            if data[feature_index] > mean_value:
                  dataset_greater.append(data)
                  label_greater.append(data[-1])
            else:
                  dataset_less.append(data)
                  label_less.append(data[-1])
    return dataset_less,dataset_greater,label_less,label_greater

def cal_entropy(dataset):
    '''
    計算數據集的熵大小
    '''
    n = len(dataset)    
    label_count = {}
    for data in dataset:
        label = data[-1]
        if label_count.has_key(label):
            label_count[label] += 1
        else:
            label_count[label] = 1
    entropy = 0
    for label in label_count:
        prob = float(label_count[label])/n
        entropy -= prob*log(prob,2)
    #print 'entropy:',entropy
    return entropy

def cal_info_gain(dataset,feature_index,base_entropy,fs_num):
    '''
    計算指定特徵對數據集的信息增益值
    g(D,F) = H(D)-H(D/F) = entropy(dataset) - sum{1,k}(len(sub_dataset)/len(dataset))*entropy(sub_dataset)
    @base_entropy = H(D)
    '''
    datasets = []
    for data in dataset:
        data = [float(n) for n in data[0:fs_num]]
        datasets.append(data)

    mean_value = mean(datasets,axis = 0)[feature_index]    #計算指定特徵的所有數據集值的平均值
    #print mean_value
    dataset_less = []
    dataset_greater = []
    for data in dataset:
        if data[feature_index] > mean_value:
            dataset_greater.append(data)
        else:
            dataset_less.append(data)
    #條件熵 H(D/F)
    condition_entropy = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
    #print 'info_gain:',base_entropy - condition_entropy
    return base_entropy - condition_entropy 

def cal_info_gain_ratio(dataset,feature_index,fs_num):
    '''
    計算信息增益比 gr(D,F) = g(D,F)/H(D)
    '''    
    base_entropy = cal_entropy(dataset)
    '''
    if base_entropy == 0:
        return 1
    '''
    info_gain = cal_info_gain(dataset,feature_index,base_entropy,fs_num)
    info_gain_ratio = info_gain/base_entropy
    return info_gain_ratio
    
def choose_best_fea_to_split(dataset,features,fs_num):
    '''
    根據每個特徵的信息增益比大小，返回最佳劃分數據集的特徵索引
    '''
    #base_entropy = cal_entropy(dataset)
    split_fea_index = -1
    max_info_gain_ratio = 0.0
    for i in range(len(features)):
        #info_gain = cal_info_gain(dataset,i,base_entropy)
        #info_gain_ratio = info_gain/base_entropy
        info_gain_ratio = cal_info_gain_ratio(dataset,i,fs_num)
        if info_gain_ratio > max_info_gain_ratio:
            max_info_gain_ratio = info_gain_ratio
            split_fea_index = i
    return split_fea_index

def most_occur_label(labels):
    '''
    返回數據集中出現次數最多的label
    '''
    label_count = {}
    for label in labels:
            if label not in label_count.keys():
                  label_count[label] = 1
            else:
                  label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sorted_label_count[0][0]

    
def build_tree(dataset,labels,features,fs_num,feature_select_list):
    '''
    創建決策樹
@dataset:訓練數據集
@labels:數據集中包含的所有label(可重複)
@features:可進行劃分的特徵集
    '''
   
    #若數據集為空,返回NULL
    if len(labels) == 0:
        return 'NULL'
    #若數據集中只有一種label,返回該label
    if len(labels) == len(labels[0]):
        return labels[0]
    #若沒有可劃分的特徵集,則返回數據集中出現次數最多的label
    if len(features) == 0:
        return most_occur_label(labels)
    #若數據集趨於穩定，則返回數據集中出現次數最多的label
    if cal_entropy(dataset) == 0:
        return most_occur_label(labels)
    split_feature_index = choose_best_fea_to_split(dataset,features,fs_num)
    split_feature = features[split_feature_index]  #split_feature 所要找到分割節點，存成列表，為feature selection所選的feature 
    decesion_tree = {split_feature:{}}
    #若劃分特徵的信息增益比小於閾值,則返回數據集中出現次數最多的label (這裡的門檻值要再想想)
    if cal_info_gain_ratio(dataset,split_feature_index,fs_num) < 0:
        return most_occur_label(labels)
    del(features[split_feature_index])
    feature_select_list.append(split_feature)
    dataset_less,dataset_greater,labels_less,labels_greater = split_dataset(dataset,split_feature_index,labels,fs_num)
    decesion_tree[split_feature]['<=']= build_tree(dataset_less,labels_less,features,fs_num,feature_select_list)
    decesion_tree[split_feature]['>']= build_tree(dataset_greater,labels_greater,features,fs_num,feature_select_list)
    return decesion_tree
    


def store_tree(decesion_tree,filename):
    '''
    把決策樹以二進制格式寫入文件
    '''
    import pickle
    writer = open(filename,'w')
    pickle.dump(decesion_tree,writer)
    writer.close()

def read_tree(filename):
    '''
    從文件中讀取決策樹，返回決策樹
    '''
    import pickle
    reader = open(filename,'rU')
    return pickle.load(reader)

def classify(decesion_tree,features,test_data,mean_values):
    '''
    對測試數據進行分類, decesion_tree : {'petal_length': {'<=': {'petal_width': {'<=': 'Iris-setosa', '>': {'sepal_width': {'<=' : 'Iris-versicolor', '>': {'sepal_length': {'<=': 'Iris-setosa', '>': 'Iris-versicolor'}}}}}}, '>': 'Iris-virginica'}} 
    '''
    first_fea = decesion_tree.keys()[0]
    fea_index = features.index(first_fea)
    if test_data[fea_index] <= mean_values[fea_index]:
        sub_tree = decesion_tree[first_fea]['<=']
        if type(sub_tree) == dict:
            return classify(sub_tree,features,test_data,mean_values)
        else:
            return sub_tree
    else:
        sub_tree = decesion_tree[first_fea]['>']
        if type(sub_tree) == dict:
            return classify(sub_tree,features,test_data,mean_values)
        else:
            return sub_tree

def get_means(train_dataset,fs_num):
    '''
    獲取訓練數據集各個屬性的數據平均值
    '''
    dataset = []
    for data in train_dataset:
        dataset.append(data[0:fs_num])
    mean_values = mean(dataset,axis = 0)   #數據集在該特徵項的所有取值的平均值
    return mean_values

def C45_main(train_file,test_file):
    '''
    主函数
    '''
    labels = get_labels(train_file)
    train_dataset,train_features,fs_num = format_data(train_file)
    dictionary={}
    for (i, data) in enumerate(train_features):
        #print "Index %d = %s" % (i, data)
        dictionary.setdefault(data,i)
    feature_select_list = []
    decesion_tree= build_tree(train_dataset,labels,train_features,fs_num,feature_select_list)
    print 'feature_select_list :',feature_select_list  #feature_select_list 為我們要的feature selection result 
    print 'decesion_tree :',decesion_tree
    #利用dictionary 和 feature_select_list 回傳index，找到對應的訓練資料與測試資料
    fs_list_index = []
    for row in feature_select_list:
        fs_list_index.append(dictionary[row])
   
    return fs_list_index
    #建立出feature selection結果的矩陣
    
    #---因為單純用C4.5做feature selection ，所以不用計算accuracy(分類才需要)
    # store_tree(decesion_tree,'decesion_tree')
    # mean_values = get_means(train_dataset,fs_num)
    # test_dataset,test_features,fs_num = format_data(test_file)
    # n = len(test_dataset)
    # correct = 0
    # for test_data in test_dataset:
        # label = classify(decesion_tree,test_features,test_data,mean_values)
        # # # #print 'classify_label  correct_label:',label,test_data[-1]
        # if label == test_data[-1]:
            # correct += 1
    # print "C4.5accuracy:",correct/float(n)

