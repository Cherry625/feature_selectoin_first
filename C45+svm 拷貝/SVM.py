import numpy as np
def SVM_main(traindata,testdata):
    import numpy as np
    from sklearn import cross_validation
    #from sklearn import svm
    import C45_classification.decision_tree
    
    #feature_select_list
    feature_select_index_list=C45_classification.decision_tree.C45_main(traindata,testdata)
    n=traindata.shape[1]
    n = n - 1
 
    
    X_train =traindata[:,feature_select_index_list]
    y_train = traindata[:,n]
    
    
    X_test =testdata[:,feature_select_index_list]
    y_test = testdata[:,n]

    #clf = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
    #accuracy = clf.score(X_test, y_test)
    
    #print "SVM Algorithm accuracy %s" % accuracy
    #return  accuracy,feature_select_index_list
    
    return feature_select_index_list 
    
