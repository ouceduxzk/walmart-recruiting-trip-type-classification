import pandas as pd 
import os
import numpy as np
a = pd.read_csv('best_0.638.csv')
b = pd.read_csv('nn_metabagging_new.csv')
c = pd.read_csv('nn_metabaggingP2.csv')
d = pd.read_csv('xgb_bagging10_new.csv')
e = pd.read_csv('nn_metabagging_0.659.csv')
f = pd.read_csv('merge10_0.648.csv')
names = a.columns[1:]
d[names] = d[names] * 0.4 + e[names] * 0.3 + f[names] * 0.3
b[names] = b[names] * 0.5 + c[names] * 0.5
a[names] = a[names] * 0.25 + b[names] * 0.45 + d[names] * 0.3
#a[names] = a[names] * 0.75 + b[names] * 0.25
a.to_csv('merge_all.csv', index=False)


# #q[names] = h[names] * 0.3 + k[names] * 0.2 + n[names] * 0.3  + o[names] * 0.2
# q[names] = h[names] * 0.2 + n[names] * 0.2 + k[names] * 0.2 + m[names] * 0.2  + o[names] * 0.2

# g[names] = g[names] * 0.7 + q[names] * 0.3
# files = os.listdir('./best')
# results = [ pd.read_csv('./best/' + x) for x in files if x[-4:]=='.csv']
# names = results[1].columns[1:]
# g = results[0]
# for i in range(1,len(results)) :
# 	g[names] = g[names] + results[i][names]

# x1 = pd.read_csv('nn_lb0.686.csv')
# x2 = pd.read_csv('nn_lb0.699.csv')
# x3 = pd.read_csv('keras_cv0.629.csv')
# x1[names] = x1[names] * 0.6 + x2[names] * 0.2 + x3[names] * 0.2

# g[names] = g[names]/len(results)
# g[names] = g[names] * 0.7 + x1[names] * 0.3

# g.to_csv('merge5xgb_3nn.csv', index=False)

