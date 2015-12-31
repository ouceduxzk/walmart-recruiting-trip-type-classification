import pandas as pd 
from  sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
import nltk
import numpy as np 
import gensim, random
import xgboost as xgb

import cPickle,re, os
from goldenFeature import *
import operator
from matplotlib import pylab as plt

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
dim = 100
weekDcit = {"Monday":1, "Tuesday" :2, "Wednesday" :3 , "Thursday" :4, "Friday" :5, "Saturday":6, "Sunday":7 }
isWeekday = {"Monday":0, "Tuesday" :0, "Wednesday" :0 , "Thursday" :0, "Friday" :0, "Saturday":1, "Sunday":1}
disWeekend = {"Monday":1, "Tuesday" :2, "Wednesday" :3 , "Thursday" :2, "Friday" :1, "Saturday":0, "Sunday":0}
seed = 1
random.seed(seed)
def read_vec(filename):
    f = open(filename, 'rb').readlines()
    wordvecs = {}
    for line in f : 
        words = line.strip().split()
        vecs = np.array(words[1:], dtype = np.float32)
        wordvecs[words[0]] = vecs
    return wordvecs

#model = gensim.models.Word2Vec.load_word2vec_format("glove.6B.100d.txt", binary=False)
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def train_with_val():
    if os.path.isfile('dataset.pkl'):
        with open('dataset.pkl', 'rb') as fp:
            newtrain = cPickle.load(fp)
            newtest = cPickle.load(fp)
            newval = cPickle.load(fp)
            train_Y = cPickle.load(fp)
            val_Y = cPickle.load(fp)
    else :
        print('--- reading input files')
        train = pd.read_csv('../../input/train.csv')
        test  = pd.read_csv('../../input/test.csv')
        sub   = pd.read_csv('../../input/sample_submission.csv')
        
        # train = train.fillna(0)
        # test = test.fillna(0)
        # train = DataFrameImputer().fit_transform(train)
        # test = DataFrameImputer().fit_transform(test)

        train['Upc'] = train['Upc'].apply(lambda x : int(str(x)[0]))
        test['Upc'] = test['Upc'].apply(lambda x : int(str(x)[0]))
        #val = val.fillna(0);

        train_y = train['TripType']
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        train_Y = le.fit_transform(train_y)
        train['TripType'] = train_Y
        
        test['tfnl1'] = test['FinelineNumber'].apply(lambda x : str(int(x) %100 + 1) )
        test['tfnl2'] = test['FinelineNumber'].apply(lambda x : str(int(x) /100 + 1) )
        train['tfnl1'] = train['FinelineNumber'].apply(lambda x : str(int(x) %100 + 1) )
        train['tfnl2'] = train['FinelineNumber'].apply(lambda x : str(int(x) /100 + 1) )
        ############################################################
        train['DDWeek'] = train['DepartmentDescription'].map(str) + train['Weekday'].map(str) 
        test['DDWeek']  = test['DepartmentDescription'].map(str) + test['Weekday'].map(str)

        train['DDScan'] = train['DepartmentDescription'].map(str) + train['ScanCount'].map(str) 
        test['DDScan']  = test['DepartmentDescription'].map(str) + test['ScanCount'].map(str)

        train['DDFln1'] = train['DepartmentDescription'].map(str) + train['tfln1'].map(str) 
        test['DDFln1']  = test['DepartmentDescription'].map(str) + test['tfln1'].map(str)
        
        train['DDFln2'] = train['DepartmentDescription'].map(str) + train['tfln2'].map(str) 
        test['DDFln2']  = test['DepartmentDescription'].map(str) + test['tfln2'].map(str)
        

        train['DDUpc'] = train['DepartmentDescription'].map(str) + train['Upc'].map(str) 
        test['DDUpc']  = test['DepartmentDescription'].map(str) + test['Upc'].map(str)


        print('add golden features with 1, 2 level of interaction ')
        golden_list = ['DDScan', 'DDFln1', 'DDFln2',  'DDWeek', 'DDUpc','DepartmentDescription']
       
        print('split the data first')
        offset1 = 520047
        train, val =  train[:offset1], train[offset1:]
        train_Y, val_Y = train_Y[:offset1], train_Y[offset1:]

        print('---loading PCA 50d of FinelineNumber')
        from sklearn.externals import joblib
        train_fln = joblib.load('pca_50_selected_compressed/train_pca_50_selected_fl_high_compressed.jl', 'rb') 
        test_fln = joblib.load('pca_50_selected_compressed/test_pca_50_selected_fl_high_compressed.jl', 'rb') 
          
        train_fln = pd.concat([train['VisitNumber'], pd.DataFrame(train_fln)], axis =1 )
        train_fln = train_fln.groupby('VisitNumber').mean()
        test_fln = pd.concat([test['VisitNumber'], pd.DataFrame(test_fln)], axis =1 )
        test_fln = test_fln.groupby('VisitNumber').mean()
        newtrain = np.hstack([newtrain, train_fln])
        newtest = np.hstack([newtest,  test_fln])
        ############################################################
        print('add golden features with 1, 2 level of interaction ')
        #golden_list = ['DepartmentDescription', 'FinelineNumber', 'Upc']
        train, val, test = getMultiLogFeatures(train, val , test, golden_list, 'TripType')
       
        train = train.drop(['DDWeek', 'DDFln1', 'DDFln2', 'DDScan', 'DDUpc'], axis = 1)
        test = test.drop(['DDWeek', 'DDFln1', 'DDFln2','DDScan', 'DDUpc'], axis = 1)
        val = val.drop(['DDWeek', 'DDFln1', 'DDFln2','DDScan', 'DDUpc'], axis = 1)

        ###########################################################
        test_week = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : weekDcit[x])
        test_isweek = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : isWeekday[x])
        test_unique_items = test.groupby('VisitNumber')['Upc'].apply(set).apply(len)
        test_return_count=  test.groupby('VisitNumber')['ScanCount'].apply(list).apply(lambda x : x.count(-1))

        #test_file = test.groupby('VisitNumber')['FinelineNumber'].median()
        test_desp_len = test.groupby('VisitNumber')['DepartmentDescription'].apply(list).apply(len)
        test_desp_uni = test.groupby('VisitNumber')['DepartmentDescription'].apply(set).apply(len)
        test_ratio =  test_desp_len/1.0/test_desp_uni
        test_row_count = test.groupby('VisitNumber')['Weekday'].count()
        test_dis_weekend = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : disWeekend[x])
        #test_row_scan_ratio = ((test_row_count.fillna(1)/test_scan.fillna(1))+ 1e-12)

        collist = test.columns.values.tolist()[2:]
        test_log = test.groupby('VisitNumber')[collist].mean()
        test_log_sum = test.groupby('VisitNumber')[collist].sum()

        train_week = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : weekDcit[x])
        train_isweek = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : isWeekday[x])
     
        train_unique_items =  train.groupby('VisitNumber')['Upc'].apply(set).apply(len)
        train_return_count= train.groupby('VisitNumber')['ScanCount'].apply(list).apply(lambda x : x.count(-1))

        #train_file = train.groupby('VisitNumber')['FinelineNumber'].median()
        train_desp_len = train.groupby('VisitNumber')['DepartmentDescription'].apply(list).apply(len)
        train_desp_uni = train.groupby('VisitNumber')['DepartmentDescription'].apply(set).apply(len)
        train_ratio = train_desp_len/1.0/train_desp_uni
        train_row_count = train.groupby('VisitNumber')['Weekday'].count()
        train_dis_weekend = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : disWeekend[x])
        #train_row_scan_ratio = ((train_row_count.fillna(1)/train_scan.fillna(1))+ 1e-12)

        collist = train.columns.values.tolist()[3:]
        train_log = train.groupby('VisitNumber')[collist].mean()
        train_log_sum = train.groupby('VisitNumber')[collist].sum()

        val_week = val.groupby('VisitNumber')['Weekday'].first().apply(lambda x : weekDcit[x])
        val_isweek = val.groupby('VisitNumber')['Weekday'].first().apply(lambda x : isWeekday[x])

        val_unique_items =  val.groupby('VisitNumber')['Upc'].apply(set).apply(len)
        val_return_count= val.groupby('VisitNumber')['ScanCount'].apply(list).apply(lambda x : x.count(-1))

        val_desp_len = val.groupby('VisitNumber')['DepartmentDescription'].apply(list).apply(len)
        val_desp_uni = val.groupby('VisitNumber')['DepartmentDescription'].apply(set).apply(len)
        
        val_ratio = val_desp_len/1.0/val_desp_uni
        val_row_count = val.groupby('VisitNumber')['Weekday'].count()
        val_dis_weekend = val.groupby('VisitNumber')['Weekday'].first().apply(lambda x : disWeekend[x])
        #val_row_scan_ratio = ((val_row_count.fillna(1)/val_scan.fillna(1))+ 1e-12)

        collist = val.columns.values.tolist()[2:]
        val_log = val.groupby('VisitNumber')[collist].mean()
        val_log_sum = val.groupby('VisitNumber')[collist].sum()

        newtest = pd.concat([test_week, test_isweek, test_unique_items,  test_desp_len , test_desp_uni, test_row_count, test_dis_weekend, test_return_count], axis = 1)
        newtrain = pd.concat([train_week, train_isweek, train_unique_items,  train_desp_len, train_desp_uni, train_row_count, train_dis_weekend, train_return_count], axis = 1)
        newval = pd.concat([ val_week, val_isweek, val_unique_items,  val_desp_len , val_desp_uni, val_row_count, val_dis_weekend, val_return_count], axis = 1)

        newtest.columns = ['week', 'isweek', 'scan','desp_len', 'desp_uni', 'row_count', 'disWeekend', 'return_scan']
        newval.columns = ['week', 'isweek', 'scan','desp_len', 'desp_uni', 'row_count', 'disWeekend', 'return_scan']
        newtrain.columns = ['week', 'isweek', 'scan','desp_len', 'desp_uni', 'row_count', 'disWeekend', 'return_scan']
       
        newtest  = pd.concat([newtest,  test_log], axis =1 )
        newval   = pd.concat([newval,    val_log], axis =1 )
        newtrain = pd.concat([newtrain,train_log], axis =1 )
        
        print newtrain.columns
        # newtrain['TripType'] = train.groupby('VisitNumber')['TripType'].first()
        # golden_list = ['isweek', 'scan' ,'disWeekend']
        # newtrain, newval, newtest = getMultiLogFeatures(newtrain, newval , newtest, golden_list, 'TripType')
        # newtrain = np.hstack( [np.array(newtrain)])
        # newval = np.hstack( [ np.array(newval) ])
        # newtest = np.hstack([ np.array(newtest)])

        train_Y = train.groupby('VisitNumber')['TripType'].first()
        val_Y   = val.groupby('VisitNumber')['TripType'].first()
        with open('dataset.pkl', 'wb') as fp:
            cPickle.dump(newtrain, fp, -1)
            cPickle.dump(newtest,  fp, -1)
            cPickle.dump(newval, fp, -1)
            cPickle.dump(train_Y,  fp, -1)
            cPickle.dump(val_Y, fp, -1)

    ##########################################################
    print('--- train a xgboost model')
    #train, val, train_label, val_label = train_test_split(newtrain, train_Y, test_size = 0.2)
    xgtrain = xgb.DMatrix(newtrain, label=train_Y)
    xgval =  xgb.DMatrix(newval, label = val_Y)
    xgtest = xgb.DMatrix(newtest)
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    params = {}
    params["objective"] = 'multi:softprob'
    params["eta"] = 0.2
    params["subsample"] = 0.66
    params["colsample_bytree"] = 0.75
    params["silent"] = 1
    params["max_depth"] = 6
    params["min_child_weight"] = 15
    params["gamma"] = 1
    params["num_class"] = len(list(set(train_Y)))
    params["eval_metric"] = 'mlogloss'
    params["seed"] = seed
    model = xgb.train(list(params.items()), xgtrain, 300, watchlist, early_stopping_rounds= 5)
    pred = model.predict(xgtest)

    importance = model.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    with open('importance.txt', 'wb') as fp:
        for x in importance :
            fp.write(x[0] + ' ' + str(x[1]) + '\n')

    sub   = pd.read_csv('../../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(pred, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('xgb_golden.csv', index=False)

def train_without_val():
    if os.path.isfile('datasets2.pkl'):
        with open('datasets2.pkl', 'rb') as fp:
            newtrain = cPickle.load(fp)
            train_Y = cPickle.load(fp)
            newtest = cPickle.load(fp)
        fp.close()
    else :
        print('--- reading input files')
        train = pd.read_csv('../../input/train.csv')
        test  = pd.read_csv('../../input/test.csv')
        sub   = pd.read_csv('../../input/sample_submission.csv')
        # train = DataFrameImputer().fit_transform(train)
        # test = DataFrameImputer().fit_transform(test)
        train = train.fillna(0)
        test = test.fillna(0)
        train['Upc'] = train['Upc'].apply(lambda x : int(str(x)[0]))
        test['Upc'] = test['Upc'].apply(lambda x : int(str(x)[0]))
        train_y = train['TripType']
        yle = LabelEncoder()
        train_Y = yle.fit_transform(train_y)
        trip_le = LabelEncoder()
        train['TripType'] = trip_le.fit_transform(train['TripType'])

        test['tfln1'] = test['FinelineNumber'].apply(lambda x : str(int(x) %100 + 1) )
        test['tfln2'] = test['FinelineNumber'].apply(lambda x : str(int(x) /100 + 1) )
        train['tfln1'] = train['FinelineNumber'].apply(lambda x : str(int(x) %100 + 1) )
        train['tfln2'] = train['FinelineNumber'].apply(lambda x : str(int(x) /100 + 1) )
        
        ############################################################
        train['DDWeek'] = train['DepartmentDescription'].map(str) + train['Weekday'].map(str) 
        test['DDWeek']  = test['DepartmentDescription'].map(str) + test['Weekday'].map(str)

        train['DDScan'] = train['DepartmentDescription'].map(str) + train['ScanCount'].map(str) 
        test['DDScan']  = test['DepartmentDescription'].map(str) + test['ScanCount'].map(str)

        train['DDFln1'] = train['DepartmentDescription'].map(str) + train['tfln1'].map(str) 
        test['DDFln1']  = test['DepartmentDescription'].map(str) + test['tfln1'].map(str)
        
        train['DDFln2'] = train['DepartmentDescription'].map(str) + train['tfln2'].map(str) 
        test['DDFln2']  = test['DepartmentDescription'].map(str) + test['tfln2'].map(str)

        train['DDUpc'] = train['DepartmentDescription'].map(str) + train['Upc'].map(str) 
        test['DDUpc']  = test['DepartmentDescription'].map(str) + test['Upc'].map(str)

        ##--------------------------------------------------------------------------------
        # test['isweekend'] = test['Weekday'].apply(lambda x : isWeekday[x])
        # test['dis_weekend'] = test['Weekday'].apply(lambda x : disWeekend[x])

        # train['isweekend'] = train['Weekday'].apply(lambda x : isWeekday[x])
        # train['dis_weekend'] = train['Weekday'].apply(lambda x : disWeekend[x])
        
        # train['DDUpf1'] = train['Upc'].map(str) + train['tfln1'].map(str) + train['DepartmentDescription'].map(str)
        # test['DDUpf1'] = test['Upc'].map(str) + test['tfln1'].map(str) + test['DepartmentDescription'].map(str)
        # train['DDUpf2'] = train['Upc'].map(str) + train['tfln2'].map(str) + train['DepartmentDescription'].map(str)
        # test['DDUpf2'] = test['Upc'].map(str) + test['tfln2'].map(str) + test['DepartmentDescription'].map(str)

        # train['DDUpcW'] = train['Upc'].map(str) + train['Weekday'].map(str) + train['DepartmentDescription'].map(str)
        # test['DDUpcW'] = test['Upc'].map(str) + test['Weekday'].map(str) + test['DepartmentDescription'].map(str)

        train['Upf1'] = train['Upc'].map(str) + train['tfln1'].map(str) 
        test['Upf1'] = test['Upc'].map(str) + test['tfln1'].map(str) 
        train['Upf2'] = train['Upc'].map(str) + train['tfln2'].map(str)
        test['Upf2'] = test['Upc'].map(str) + test['tfln2'].map(str)

        #--------------------------------------------------------------------------------------------------------------------------------

        print('add golden features with 1, 2 level of interaction ')
        golden_list = ['Upf1', 'Upf2', 'DDScan', 'DDFln1', 'DDFln2',  'DDWeek', 'DDUpc','DepartmentDescription']
        train, test = getMultiLogFeaturesNoVal(train, test, golden_list, 'TripType')

        train = train.drop(['Upf1', 'Upf2','DDWeek', 'DDFln1', 'DDFln2', 'DDScan', 'DDUpc'], axis = 1)
        test = test.drop([ 'Upf1', 'Upf2', 'DDWeek',  'DDFln1', 'DDFln2','DDScan', 'DDUpc'], axis = 1)
        
        #---------------------------------------------------------------------------------------------------------------------------------
        test_week = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : weekDcit[x])
        test_isweek = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : isWeekday[x])
        test_scan = test.groupby('VisitNumber')['ScanCount'].sum()
        test_return_scan = test.groupby('VisitNumber')['ScanCount'].apply(list).apply(lambda x : 1 if -1 in x else 0)

        #test_fln1 = test.groupby('VisitNumber')['FinelineNumber'].apply(list).apply(lambda x : np.mean([int(a) % 100 for a in x]))
        #test_fln2 = test.groupby('VisitNumber')['FinelineNumber'].apply(list).apply(lambda x : np.mean([int(a) / 100 for a in x]))
        #test_file = test.groupby('VisitNumber')['FinelineNumber'].median()
        test_desp_len = test.groupby('VisitNumber')['DepartmentDescription'].apply(list).apply(len)
        test_desp_uni = test.groupby('VisitNumber')['DepartmentDescription'].apply(set).apply(len)
        test_ratio =  test_desp_len/test_desp_uni
        test_row_count = test.groupby('VisitNumber')['Weekday'].count()
        test_dis_weekend = test.groupby('VisitNumber')['Weekday'].first().apply(lambda x : disWeekend[x])
        test_row_scan_ratio = ((test_row_count.fillna(1)/test_scan.fillna(1))+ 1e-12)

        collist = test.columns.values.tolist()[5:]
        test_log = test.groupby('VisitNumber')[collist].mean()
        test_log_sum = test.groupby('VisitNumber')[collist].sum()

        train_week = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : weekDcit[x])
        train_isweek = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : isWeekday[x])
        train_scan = train.groupby('VisitNumber')['ScanCount'].sum()
        train_return_scan = train.groupby('VisitNumber')['ScanCount'].apply(list).apply(lambda x : 1 if -1 in x else 0)

        #train_fln1 = train.groupby('VisitNumber')['FinelineNumber'].apply(list).apply(lambda x : np.mean([int(a) % 100 for a in x]))
        #train_fln2 = train.groupby('VisitNumber')['FinelineNumber'].apply(list).apply(lambda x : np.mean([int(a) / 100 for a in x]))

        #train_file = train.groupby('VisitNumber')['FinelineNumber'].median()
        train_desp_len = train.groupby('VisitNumber')['DepartmentDescription'].apply(list).apply(len)
        train_desp_uni = train.groupby('VisitNumber')['DepartmentDescription'].apply(set).apply(len)
        train_ratio = train_desp_len/train_desp_uni
        train_row_count = train.groupby('VisitNumber')['Weekday'].count()
        train_dis_weekend = train.groupby('VisitNumber')['Weekday'].first().apply(lambda x : disWeekend[x])
        train_row_scan_ratio = ((train_row_count.fillna(1)/train_scan.fillna(1))+ 1e-12)

        collist = train.columns.values.tolist()[6:]
        train_log = train.groupby('VisitNumber')[collist].mean()
        train_log_sum = train.groupby('VisitNumber')[collist].sum()

        newtest = pd.concat([ test_week, test_isweek, test_scan,  test_desp_len , test_desp_uni, test_row_count, test_dis_weekend, test_return_scan], axis = 1)
        newtrain = pd.concat([ train_week, train_isweek, train_scan,  train_desp_len, train_desp_uni, train_row_count, train_dis_weekend, train_return_scan], axis = 1)

        newtest.columns = ['week', 'isweek', 'scan','desp_len', 'desp_uni', 'row_count', 'disWeekend', 'return_scan']
        newtrain.columns =['week', 'isweek', 'scan','desp_len', 'desp_uni', 'row_count', 'disWeekend', 'return_scan']

        newtest = pd.concat([newtest, test_log], axis =1 )
        newtrain = pd.concat([newtrain,train_log ], axis =1 )
        ceate_feature_map(newtrain.columns)

        # newtrain = newtrain.fillna(-1)
        # newtest = newtest.fillna(-1)
        # print newtrain.columns   
        train_Y = train.groupby('VisitNumber')['TripType'].first()
        print('---loading PCA 50d of FinelineNumber')
        from sklearn.externals import joblib
        train_fln = joblib.load('pca_50_selected_compressed/train_pca_50_selected_fl_high_compressed.jl', 'rb') 
        test_fln = joblib.load('pca_50_selected_compressed/test_pca_50_selected_fl_high_compressed.jl', 'rb') 
          
        train_fln = pd.concat([train['VisitNumber'], pd.DataFrame(train_fln)], axis =1 )
        train_fln = train_fln.groupby('VisitNumber').mean()
        test_fln = pd.concat([test['VisitNumber'], pd.DataFrame(test_fln)], axis =1 )
        test_fln = test_fln.groupby('VisitNumber').mean()
        newtrain = np.hstack([newtrain, train_fln])
        newtest = np.hstack([newtest,  test_fln])
        with open('datasets2.pkl', 'wb') as fp:
            cPickle.dump(newtrain, fp, -1)
            cPickle.dump(train_Y,  fp, -1)
            cPickle.dump(newtest,  fp, -1)
        fp.close()

    ##########################################################
    ########### maybe put some golden feature here.
    ##########################################################
    print('---train a xgboost model')
    train = pd.read_csv('../../input/train.csv')
    test  = pd.read_csv('../../input/test.csv')
    # newtrain = np.array( newtrain, dtype = np.float32)
    # newtest  = np.array( newtest,  dtype = np.float32)
    # train_Y = np.array(  train_Y,  dtype = np.int32)
    #---------- dummy of dd-----------------------------------------
    # train_dd = pd.get_dummies(train['DepartmentDescription'])
    # train_dd = pd.concat([train['VisitNumber'], train_dd], axis=1)
    # train_dd = train_dd.groupby('VisitNumber').sum()
    # test_dd = pd.get_dummies(test['DepartmentDescription'])
    # test_dd = pd.concat([train['VisitNumber'], test_dd], axis=1)
    # test_dd = test_dd.groupby('VisitNumber').sum()
    # print train_dd.shape
    # newtest = np.hstack([newtest, test_dd])
    # newtrain =np.hstack([newtrain,train_dd ])
    #-----------------------------------------------------------------

    print newtrain.shape
    train, val,  train_label, val_label = train_test_split(newtrain, train_Y, test_size = 0.1)

    train = np.array( train, dtype = np.float32)
    val = np.array(val, dtype = np.float32)
    test = np.array(newtest, dtype = np.float32)
    train_label  = np.array( train_label,  dtype = np.int32)
    val_label = np.array(  val_label,  dtype = np.int32)

    xgtrain = xgb.DMatrix(train, label=train_label)
    xgval = xgb.DMatrix(val,     label=val_label)
    xgtest  = xgb.DMatrix(test)
    params  = {}
    params["objective"] = 'multi:softprob'
    params["eta"] = 0.1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.55
    params["silent"] = 1
    params["max_depth"] = 6
    params["min_child_weight"] = 10
    #params["gamma"] = 2
    params["num_class"] = len(list(set(train_Y)))
    params["eval_metric"] = 'mlogloss'
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(list(params.items()), xgtrain, 120, watchlist, early_stopping_rounds = 5)
    pred = model.predict(xgtest)
    sub   = pd.read_csv('../../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(pred, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('xgb_last.csv', index=False)

'''
    params["objective"] = 'multi:softprob'
    params["eta"] = 0.1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.55
    params["silent"] = 1
    params["max_depth"] = 8
    params["min_child_weight"] = 12
    params["gamma"] = 1
    100 rounds, got 0.688 lb , cv 0.624.

    params["objective"] = 'multi:softprob'
    params["eta"] = 0.1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.65
    params["silent"] = 1
    params["max_depth"] = 6
    params["min_child_weight"] = 12 ,cv 0.604, lb 0.666

    params["objective"] = 'multi:softprob'
    params["eta"] = 0.1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.65
    params["silent"] = 1
    params["max_depth"] = 6
    params["min_child_weight"] = 10 ,cv 0.596, lb ?

'''
def tfidf_noval():
    print('--- reading input files')
    train = pd.read_csv('../../input/train.csv')
    test  = pd.read_csv('../../input/test.csv')
    sub   = pd.read_csv('../../input/sample_submission.csv')
    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)
    print('--- add TFIDF features ')
    train_desp_list = []
    test_desp_list = []
    for lists in test.groupby('VisitNumber')['DepartmentDescription'].apply(list):
        try :
                lists = [x for x in lists if x not in stopwords.words('english') and  str(x) != 'nan']
                lists = re.sub('[^a-zA-Z]+', ' ', ' '.join(lists)).strip().lower().split()
                test_desp_list.append(' '.join(lists))
        except TypeError:
                test_desp_list.append(['None'])
                continue

    for lists in train.groupby('VisitNumber')['DepartmentDescription'].apply(list):
        try:
                lists = [x for x in lists if x not in stopwords.words('english') and str(x) != 'nan']
                lists = re.sub('[^a-zA-Z]+', ' ', ' '.join(lists)).strip().lower().split()
                train_desp_list.append(' '.join(lists))
        except TypeError:
                test_desp_list.append(['None'])
                continue
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vect = TfidfVectorizer(ngram_range=(1,1), min_df=0, max_features = 1000) #analyzer
    train_desp_vec = vect.fit_transform(train_desp_list)
    test_desp_vec =  vect.transform(test_desp_list)

    vect1 = TfidfVectorizer(ngram_range=(2,13), min_df=0, analyzer='char', max_features = 2000) #analyzer
    train_desp_vec1 = vect.fit_transform(train_desp_list)
    test_desp_vec1 =  vect.transform(test_desp_list)

    train_tfidf = np.hstack([train_desp_vec.toarray() , train_desp_vec1.toarray() ])
    test_tfidf =  np.hstack([test_desp_vec.toarray(), test_desp_vec1.toarray() ])

    print train_tfidf.shape
    with open('walmart_tfidf.pkl' , 'wb') as fp:
        cPickle.dump(train_tfidf, fp, -1)
        cPickle.dump(test_tfidf, fp, -1)


def dummpyFLNPCA():
    from scipy import stats
    print('--- reading input files')
    train = pd.read_csv('../../input/train.csv')
    test  = pd.read_csv('../../input/test.csv')
    sub   = pd.read_csv('../../input/sample_submission.csv')
    # train = DataFrameImputer().fit_transform(train)
    # test = DataFrameImputer().fit_transform(test)
    # plt.hist(train['FinelineNumber'])
    # plt.show()
    train_fln = train.groupby('VisitNumber')['FinelineNumber'].apply(lambda x : stats.mode(x)[0][0]) #.agg(lambda x: stats.mode(x['FinelineNumber'])[0])
    train_fln = pd.get_dummies(train_fln)
    print train_fln.shape
    test_fln  = pd.get_dummies(test.groupby('VisitNumber')['FinelineNumber'].apply(lambda x : stats.mode(x)[0][0]))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    train_fln = pca.fit_transform(train_fln)
    print('--PCA train finished')
    test_fln = pca.fit_transform(test_fln)
    
    print train_fln
    with open('fln.pkl', 'wb') as fp:
        cPickle.dump(train_fln, fp, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_fln, fp, cPickle.HIGHEST_PROTOCOL)

train_without_val()

#dummpyFLNPCA()