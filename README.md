# Bayesian
Naive Bayes classifier and Bayesian network classifier C++ implementation

Reference :

[Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

[Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network)

Usage:
```
./bayesian [training data file] [test data file] [configuration file] [method]
```
method :
```
0 : naive bayesian
1 : bayesian network
```

Example:
```
./bayesian breast_cancer_data/breast-cancer-wisconsin-400-records-train breast_cancer_data/breast-cancer-wisconsin-299-records-test breast_cancer_data/breast-cancer-wisconsin.cfg 0
```


Training & Test data are in CSV format:
```
[ attribute 1 of data 1 ],[ attribute 2 of data 1 ], ........ ,[ true result of data 1 ]
[ attribute 1 of data 2 ],[ attribute 2 of data 2 ], ........ ,[ true result of data 2 ]
[ attribute 1 of data 3 ],[ attribute 2 of data 3 ], ........ ,[ true result of data 3 ]
```
* For real prediction, don't need to provide true result. Need to pass second argument "0" for the "Predict" API.

Configuration file format:
```
[ number of training instance ]  [ number of test instance ]  [ number of attributes ]

[ attribute 1 is discrete(0) or continunous(1) ]  [ attribute 2 is discrete(0) or continunous(1) ]  ......

[ number of types for attributes 1 ]  [ number of types for attributes 2 ]  .......[ number of classes for prediction result ]  

* For continuous data, the "number of types for attributes" won't matter.
* Bayesian Network DO NOT support continuous data for now.
```

Data set is acquired from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
