# walmart-recruiting-trip-type-classification

* 81th/1069 Solution for Kaggle walmart-recruiting-trip-type-classification

### Task ###

* Classification of 38 Trip types based on 5 features( department, Upc, FinelineNumber, Weekday, Scan count)


### Feature  Engineering###

* TFIDF of position, department descriptions 
*  Cross-entropy features of A with respect to target C
*  PCA of one-hot encoding of FinelineNumbers
* some hard-coded features 

### Models ###

* GBDT, DNN, KNN, Random Forest.


### Metabagging ###

*  Train Dataset are split into two parts A, B. 
*  Two metaclassifier : GBDT and DNN are used for the final ensemble.
*   each of the model are ensembled by 10 same models with features mentioned above plus predictions of base classifier(like rf, knn) 
*   The base classiers are trained on A and then predict on B, which will be the metafeatures for model above.