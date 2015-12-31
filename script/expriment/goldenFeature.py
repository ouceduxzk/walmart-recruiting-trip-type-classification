import pandas as pd
import numpy as  np
from copy import deepcopy
import sys , operator
def getNewLogFeatures(train, val, test,  nameA, nameC):
    logoddsCurrSalary, logoddsPCurrSalary, default_logodds_Sal = get_log_count(train, nameA, nameC)
    len_train = len(train.columns.values.tolist())
    train1 , val1 = generate_log_features(train, val, logoddsCurrSalary, logoddsPCurrSalary, nameA, nameC, default_logodds_Sal)
    train1, test1 = generate_log_features(train, test, logoddsCurrSalary, logoddsPCurrSalary, nameA, nameC, default_logodds_Sal)
    #return train1[len_train:], val1[len_train:], test1[len_train-1:]
    return train1, val1, test1

def getNewLogFeaturesNoVal(train, test,  nameA, nameC):
    logoddsCurrSalary, logoddsPCurrSalary, default_logodds_Sal = get_log_count(train, nameA, nameC)
    len_train = len(train.columns.values.tolist())
    train1, test1 = generate_log_features(train, test, logoddsCurrSalary, logoddsPCurrSalary, nameA, nameC, default_logodds_Sal)
    #return train1[len_train:], val1[len_train:], test1[len_train-1:]
    return train1, test1

def getMultiLogFeaturesNoVal(train,  test, xlist, nameC):
    i = 0
    num_features = len(train.columns)
    for nameA in xlist:
        if i == 0:
            train1, test1 = getNewLogFeaturesNoVal(train, test, nameA, nameC)
        else :
            trainX,  testX = getNewLogFeaturesNoVal(train, test, nameA, nameC) 
            keep_list = trainX.columns[num_features:]
            train1  = pd.concat([train1, trainX[keep_list] ], axis =1 )
            test1 = pd.concat([test1, testX[keep_list]], axis = 1)
        i = i + 1
    return train1, test1

def getMultiLogFeatures(train, val , test, xlist, nameC):
    i = 0
    num_features = len(train.columns)
    for nameA in xlist:
        if i == 0:
            train1, val1, test1 = getNewLogFeatures(train, val, test, nameA, nameC)
        else :
            trainX, valX, testX = getNewLogFeatures(train, val, test, nameA, nameC) 
            keep_list = trainX.columns[num_features:]
            train1  = pd.concat([train1, trainX[keep_list] ], axis =1 )
            val1 = pd.concat([val1, valX[keep_list]], axis =1 )
            test1 = pd.concat([test1, testX[keep_list]], axis = 1)
        i = i + 1
    return train1, val1, test1

def  getTwoWayInteraction(train, val , test, xlist, nameC):
    i = 0
    num_features = len(train.columns)
    for nameA, nameB in xlist:
        train[nameA+nameB] = train[nameA].map(str) + train[nameB].map(str)
        val[nameA+nameB]   = val[nameA].map(str)   +   val[nameB].map(str)
        test[nameA+nameB]  = test[nameA].map(str)  +  test[nameB].map(str)
    newlist = [ x+y for x, y in xlist ]
    print newlist
    train, val, test = getMultiLogFeatures(train, val, test, newlist, nameC)
    for nameA, nameB in xlist:
        train = train.drop(nameA+nameB, 1)
        val = val.drop(nameA+nameB, 1)
        test = test.drop(nameA+nameB,1)
    return train, val, test

def parse_data(df,logodds,logoddsPA, NameA, NameC):

    feature_list=df.columns.tolist()
    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print("Creating A features")
    address_features=cleanData[NameA].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+ NameA + NameC + str(x) for x in range(len(address_features.columns))]
    #print("Creating one-hot variables")
    #dummy_ranks_PD = pd.get_dummies(cleanData['Upc'], prefix='U')
    #dummy_ranks_DAY = pd.get_dummies(cleanData["FinelineNumber"], prefix='FN')
    cleanData["logodds" + NameA + NameC ]=cleanData[NameA].apply(lambda x: logoddsPA[x])
    #cleanData=cleanData.drop("Upc",axis=1)
    #cleanData=cleanData.drop("FinelineNumber",axis=1)
    feature_list=cleanData.columns.tolist()
    features = cleanData[feature_list].join(address_features.ix[:,:])
    return features


def get_log_count(trainDF, NameA, NameC):
    addresses=sorted(trainDF[NameA].unique())
    categories=sorted(trainDF[NameC].unique())
    C_counts=trainDF.groupby([NameC]).size()
    A_C_counts=trainDF.groupby([NameA, NameC]).size()
    A_counts=trainDF.groupby([NameA]).size()
    logodds={}
    logoddsPA={}
    MIN_CAT_COUNTS=2

    default_logodds=np.log(C_counts/len(trainDF))- np.log(1.0-C_counts/float(len(trainDF)))
    for addr in addresses:
        PA=A_counts[addr]/float(len(trainDF))
        logoddsPA[addr]=np.log(PA)- np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        for cat in A_C_counts[addr].keys():
            if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
                PA=A_C_counts[addr][cat]/float(A_counts[addr])
                #logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
                logodds[addr][cat]=np.log(PA)-np.log(1.0-PA)
        logodds[addr]=pd.Series(logodds[addr])
        logodds[addr].index=range(len(categories))
    return logodds, logoddsPA, default_logodds

def generate_log_features(trainDF, testDF, logodds, logoddsPA, NameA, NameC, default_logodds):
    addresses=sorted(trainDF[NameA].unique())
    A_counts=trainDF.groupby([NameA]).size()
    categories=sorted(trainDF[NameC].unique())
    features = parse_data(trainDF,logodds,logoddsPA, NameA, NameC)
    collist=features.columns.tolist()[2:]
    # scaler = StandardScaler()
    # scaler.fit(features[collist])
    # features[collist]=scaler.transform(features[collist])
    new_addresses=sorted(testDF[NameA].unique())
    new_A_counts=testDF.groupby(NameA).size()
    only_new=set(new_addresses+addresses)-set(addresses)
    only_old=set(new_addresses+addresses)-set(new_addresses)
    in_both=set(new_addresses).intersection(addresses)
    for addr in only_new:
        PA=new_A_counts[addr]/float(len(testDF)+len(trainDF))
        logoddsPA[addr]=np.log(PA)- np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        logodds[addr].index=range(len(categories))

    for addr in in_both:
        PA=(A_counts[addr]+new_A_counts[addr])/float(len(testDF)+len(trainDF))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    features_sub =parse_data(testDF,logodds,logoddsPA, NameA, NameC)
    # scaler.fit(features_test)
    #collist=features_sub.columns.tolist()[1:]
    #features_sub[collist]=scaler.transform(features_sub[collist])
    return features, features_sub
