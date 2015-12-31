import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
import theano
from scipy.special import expit
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
import cPickle
import xgboost as xgb
random.seed(21)



with open('datasets.pkl', 'rb') as fp:
    train = cPickle.load(fp)
    train_y = cPickle.load(fp)
    test = cPickle.load(fp) 
    from sklearn.preprocessing import StandardScaler
    sle = StandardScaler()
    sle.fit(train)
    train = sle.transform(train)
    test = sle.transform(test)

with open('walmart_tfidf.pkl', 'rb') as fp:
    train_tfidf = cPickle.load(fp)
    test_tfidf  = cPickle.load(fp)

train = np.array(train, dtype = np.float32)
test  = np.array(test, dtype = np.float32)
train_y = np.array(train_y, dtype = np.int32)

offset = 10000

LINES = train.shape[0]
print LINES


def skleanBagging(train, y, test, model):
    # Check classification for various parameter settings.
    #X_train, X_test, y_train, y_test = train_test_split(train, y)
    clf = BaggingClassifier(model,n_estimators=10,max_samples=1.0,bootstrap=True,verbose=2)
    return clf.fit(train,y).predict_proba(test)

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def xgb_train_data(train, train_y, ttf):

    num_features = train.shape[1]
    X = train.copy()
    y = np.array(train_y[offset:LINES].copy(), dtype = np.int32)
    rbm1 = KNeighborsClassifier(n_neighbors=5).fit(X[0:offset-1,:], train_y[0:offset-1])
    rbm2 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=8, verbose=1).fit(X[0:offset-1,:], train_y[0:offset-1])
    layers0 = [('input', InputLayer),
           ('dropout0', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

    rbm3 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout0_p = 0.05, #theano.shared(float32(0.1)),
                    dense0_num_units= 100,
                     dropout1_p= 0.1, #theano.shared(float32(0.5)),
                     dense1_num_units= 200,
                     dropout2_p =  0.3, #theano.shared(float32(0.8)),
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     #update_learning_rate=0.005,
                     #update_momentum=0.9,
                     update_learning_rate = theano.shared(float32(0.001)),
             update_momentum=theano.shared(float32(0.9)),
                     #objective_loss_function = log_loss,
                     train_split = TrainSplit(0.2),
                     verbose=1,
                     max_epochs=300,
                   on_epoch_finished=[
                   AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
                      AdjustVariable('update_momentum', start=0.9, stop=0.99),
                    #   AdjustDropout('dropout0_p', start = 0.1, stop = 0.2),
                     #   #AdjustDropout('dropout1_p', start = 0.5, stop = 0.4),
                     #  AdjustDropout('dropout2_p', start = 0.8, stop = 0.9)
    ])
    rbm3 = rbm3.fit(X[0:offset-1,:], train_y[0:offset-1])
    X =  np.hstack([X[offset:LINES,:], rbm1.predict_proba(X[offset:LINES,:]), rbm2.predict_proba(X[offset:LINES,:]) , rbm3.predict_proba(X[offset:LINES,:]) ] )    
    return np.array(X, dtype = np.float32), y, rbm1, rbm2, rbm3

def xgb_test_data(test, tef, rbm1, rbm2, rbm3):
    #df = pd.read_csv(path)
    #X = df.values.copy()
    X = test.copy()
    #X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    #X = scaler.transform(np.log(1+X))
    X =  np.hstack([X, rbm1.predict_proba(X), rbm2.predict_proba(X), rbm3.predict_proba(X)] ) 
    return np.array(X,dtype = np.float32)

def load_train_data(train, train_y, ttf):
    X = train.copy()
    y = np.array(train_y[offset:LINES].copy(), dtype = np.int32)
    rbm0 = ExtraTreesClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=5, n_jobs = 8).fit(X[0:offset-1,:], train_y[0:offset-1])
    #rbm1 = xgb.XGBClassifier(n_estimators=200,max_depth=6,subsample=0.8,min_child_weight = 2, nthread=8).fit(X[0:offset-1,:], train_y[0:offset-1])
    rbm1 = KNeighborsClassifier(n_neighbors = 5).fit(X[0:offset-1,:], train_y[0:offset-1])
    rbm2 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=8, verbose=1).fit(X[0:offset-1,:], train_y[0:offset-1])
    rbm3 = xgb.XGBClassifier(n_estimators=300,max_depth=8,subsample=0.8,min_child_weight=4,nthread=8).fit(X[0:offset-1,:], train_y[0:offset-1])
    X =  np.hstack([X[offset:LINES,:], rbm0.predict_proba(X[offset:LINES,:]), rbm1.predict_proba(X[offset:LINES,:]), np.power(rbm2.predict_proba(X[offset:LINES,:])*rbm3.predict_proba(X[offset:LINES,:]), (1/2.0)) ] )    
    return np.array(X, dtype = np.float32), y, rbm0, rbm1, rbm2, rbm3

def load_test_data(test, tef, rbm0, rbm1, rbm2, rbm3):
    #df = pd.read_csv(path)
    #X = df.values.copy()
    X = test.copy()
    #X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    #X = scaler.transform(np.log(1+X))
    X =  np.hstack([X, rbm0.predict_proba(X), rbm1.predict_proba(X), np.power(rbm2.predict_proba(X)*rbm3.predict_proba(X), (1/2.0))] ) 
    return np.array(X,dtype = np.float32)

def make_submission(y_prob, ids, encoder, name='/home/mikeskim/Desktop/kaggle/otto/data/lasagneSeed21.csv'):
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))


def WalmartMetaBagging(train, train_y, test, train_tfidf, test_tfidf):
  usenn = True
  usexgb = False
  num_runs = 5
  num_classes = 38

  print('1.metabagging with neural_network')

  if usenn:
    #Load Data
    X, y, rbm0, rbm1, rbm2, rbm3 = load_train_data(train, train_y, train_tfidf)
    X_test= load_test_data(test, test_tfidf, rbm0, rbm1, rbm2, rbm3)
    num_features = X.shape[1]
    print(num_classes); print(num_features); print(train)

    layers0 = [('input', InputLayer),
           ('dropout0', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout0_p = 0.05, #theano.shared(float32(0.1)),
                    dense0_num_units= 100,
                     dropout1_p= 0.1, #theano.shared(float32(0.5)),
                     dense1_num_units= 200,
                     dropout2_p =  0.3, #theano.shared(float32(0.8)),
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     #update_learning_rate=0.005,
                     #update_momentum=0.9,
                     update_learning_rate = theano.shared(float32(0.001)),
             update_momentum=theano.shared(float32(0.9)),
                     #objective_loss_function = log_loss,
                     train_split = TrainSplit(0.2),
                     verbose=1,
                     max_epochs=250,
                   on_epoch_finished=[
                   AdjustVariable('update_learning_rate', start=0.002, stop=0.0001),
                      AdjustVariable('update_momentum', start=0.9, stop=0.99),
                    #   AdjustDropout('dropout0_p', start = 0.1, stop = 0.2),
                     #   #AdjustDropout('dropout1_p', start = 0.5, stop = 0.4),
                     #  AdjustDropout('dropout2_p', start = 0.8, stop = 0.9)
              ]
    )
    print(X)
    net0.fit(X, y)
    y_prob = net0.predict_proba(X_test)

    for jj in xrange(num_runs):
      print(jj)
      XX, yy, rbm0, rbm1, rbm2, rbm3 = load_train_data(train, train_y, train_tfidf)
      XX_test = load_test_data(test, test_tfidf, rbm0, rbm1,  rbm2, rbm3)
      num_features = X.shape[1]
      net0 = NeuralNet(layers=layers0,
                       input_shape=(None, num_features),
                       dropout0_p = 0.05, #theano.shared(float32(0.1)),
                      dense0_num_units= 100,
                       dropout1_p= 0.1, #theano.shared(float32(0.5)),
                       dense1_num_units= 200,
                       dropout2_p =  0.3, #theano.shared(float32(0.8)),
                       output_num_units=num_classes,
                       output_nonlinearity=softmax,
                       update=nesterov_momentum,
                       #update_learning_rate=0.005,
                       #update_momentum=0.9,
                       update_learning_rate = theano.shared(float32(0.001)),
               update_momentum=theano.shared(float32(0.9)),
                       #objective_loss_function = log_loss,
                       train_split = TrainSplit(0.2),
                       verbose=1,
                       max_epochs=250,
                     on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.002, stop=0.0001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.99),
                      #   AdjustDropout('dropout0_p', start = 0.1, stop = 0.2),
                       #   #AdjustDropout('dropout1_p', start = 0.5, stop = 0.4),
                       #  AdjustDropout('dropout2_p', start = 0.8, stop = 0.9)
      ])
      y = np.array(y, dtype = np.int32)
      net0.fit(XX, yy)
      y_prob = y_prob + net0.predict_proba(XX_test)

    y_prob = y_prob/(num_runs+1.0)
    sub   = pd.read_csv('../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(y_prob, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('nn_metabagging.csv', index=False)

  
  num_runs = 2

  print('2. metabagging with xgboost')
  if usexgb:
    X, y,rbm1, rbm2, rbm3 = xgb_train_data(train, train_y, train_tfidf)
    X_test= xgb_test_data(test, test_tfidf, rbm1, rbm2, rbm3)
    X_train, X_val, train_label, val_label = train_test_split(train, train_y, test_size = 0.2)

    X_train = np.array( X_train, dtype = np.float32)
    X_val = np.array(X_val, dtype = np.float32)
    test = np.array(test, dtype = np.float32)
    train_label  = np.array( train_label,  dtype = np.int32)
    val_label = np.array( val_label,  dtype = np.int32)

    xgtrain = xgb.DMatrix(X_train, label=train_label)
    xgval  = xgb.DMatrix(X_val, label = val_label)
    
    ########## xtest to test
    xgtest = xgb.DMatrix(X_test)
    params = {}
    params["objective"] = 'multi:softprob'
    params["eta"] = 0.1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.55
    params["silent"] = 1
    params["max_depth"] = 8
    params["min_child_weight"] = 12
    params["gamma"] = 1
    params["num_class"] = 38
    params["eval_metric"] = 'mlogloss'
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(list(params.items()), xgtrain, 120, watchlist, early_stopping_rounds = 5)
    xgb_pred = model.predict(xgtest)

    for jj in xrange(num_runs):
      print(jj)
      # XX, yy, rbm1, rbm2, rbm3 = xgb_train_data(train, train_y, train_tfidf)
      # XX_test = xgb_test_data(test, test_tfidf, rbm1, rbm2, rbm3)
      model = xgb.train(list(params.items()), xgtrain, 120, watchlist, early_stopping_rounds = 5)
      xgb_pred += model.predict(xgtest)

    xgb_pred = xgb_pred/(num_runs+1.0)

    # from sklearn.ensemble import BaggingClassifier
    # baggs = BaggingClassifier(base_estimator=model, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)[source]

    sub   = pd.read_csv('../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(xgb_pred, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('xgb_metabagging.csv', index=False)

#WalmartMetaBagging()
def xgbBaggging():
    clf = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 6, min_child_weight=10, colsample_bytree =0.65, subsample = 0.7, objective =  'multi:softprob', gamma = 1, nthread=-1, seed = 0)
    pred =  skleanBagging(train, train_y, test, clf)
    sub   = pd.read_csv('../../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(pred, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('xgb_bagging10_new.csv', index=False)

def nnBagging():
    num_features = train.shape[1]
    layers0 = [('input', InputLayer),
               ('dropout0', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout0_p = 0.05, #theano.shared(float32(0.1)),
                    dense0_num_units= 100,
                     dropout1_p= 0.1, #theano.shared(float32(0.5)),
                     dense1_num_units= 200,
                     dropout2_p =  0.3, #theano.shared(float32(0.8)),
                     output_num_units=38,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     #update_learning_rate=0.005,
                     #update_momentum=0.9,
                     update_learning_rate = theano.shared(float32(0.001)),
                    update_momentum=theano.shared(float32(0.9)),
                     #objective_loss_function = log_loss,
                     train_split = TrainSplit(0.2),
                     verbose=1,
                     max_epochs=300,
                   on_epoch_finished=[
                   AdjustVariable('update_learning_rate', start=0.002, stop=0.0001),
                      AdjustVariable('update_momentum', start=0.9, stop=0.99),
                    #   AdjustDropout('dropout0_p', start = 0.1, stop = 0.2),
                     #   #AdjustDropout('dropout1_p', start = 0.5, stop = 0.4),
                     #  AdjustDropout('dropout2_p', start = 0.8, stop = 0.9)
              ]
    )
    pred =  skleanBagging(train, train_y, test, net0)
    sub   = pd.read_csv('../input/sample_submission.csv')
    cols = sub.columns.values.tolist()[1:]
    sub[cols] = pd.DataFrame(np.around(pred, decimals=5)).applymap(lambda x: round(x, 5))
    sub.to_csv('nn_bagging10.csv', index=False)





# layers0 = [('input', InputLayer),
#        ('dropout0', DropoutLayer),
#            ('dense0', DenseLayer),
#            ('dropout1', DropoutLayer),
#            ('dense1', DenseLayer),
#            ('dropout2', DropoutLayer),
#            ('output', DenseLayer)]

# net0 = NeuralNet(layers=layers0,
#                  input_shape=(None, train.shape[1]),
#                  dropout0_p = 0.1, #theano.shared(float32(0.1)),
#                 dense0_num_units= 200,
#                  dropout1_p= 0.2, #theano.shared(float32(0.5)),
#                  dense1_num_units= 400,
#                  dropout2_p =  0.3, #theano.shared(float32(0.8)),
#                  output_num_units=38,
#                  output_nonlinearity=softmax,
#                  update=nesterov_momentum,
#                  #update_learning_rate=0.005,
#                  #update_momentum=0.9,
#                  update_learning_rate = theano.shared(float32(0.001)),
#          update_momentum=theano.shared(float32(0.9)),
#                  #objective_loss_function = log_loss,
#                  train_split = TrainSplit(0.2),
#                  verbose=1,
#                  max_epochs=300,
#                on_epoch_finished=[
#                AdjustVariable('update_learning_rate', start=0.002, stop=0.0001),
#                   AdjustVariable('update_momentum', start=0.9, stop=0.99),
#                 #   AdjustDropout('dropout0_p', start = 0.1, stop = 0.2),
#                  #   #AdjustDropout('dropout1_p', start = 0.5, stop = 0.4),
#                  #  AdjustDropout('dropout2_p', start = 0.8, stop = 0.9)
#           ]
# )
# print(train)
# net0.fit(train, train_y)


#xgbBaggging()
#nnBaggging()
WalmartMetaBagging(train, train_y, test, train_tfidf, test_tfidf)