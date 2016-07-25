####################################################################################################
####################################################################################################
##

import pandas
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import time
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV

####################################################################################################
####################################################################################################
#####

breast_cancer_df = pandas.read_csv("../data/breast-cancer.csv", header = None)

##### dependent variable y: 1 means malignant, -1 means benign
y = numpy.where(numpy.array(breast_cancer_df.ix[:, 1]) == "M", 1, -1)
print y.shape

##### number of malignant and benign
unique, counts = numpy.unique(y, return_counts=True)
print "The dependent variable is unbalanced: ", numpy.asarray((unique, counts)).T

##### independent variables X:
X = breast_cancer_df.ix[:, 2:(breast_cancer_df.shape[1])].as_matrix()
print X.shape

##### fit logistic regression model
logistic_estimator = LogisticRegression(class_weight = 'balanced')
logistic_regression_cv_score = cross_val_score(logistic_estimator, X, y, cv = 5, n_jobs = -1)
print "LogisticRegression"
print "Mean accuracy:", numpy.mean(logistic_regression_cv_score)
print "Mean standard deviation:", numpy.std(logistic_regression_cv_score)

##### fit random forest model
#random_forest_estimator = RandomForestClassifier(class_weight = 'balanced', n_estimators = 30)
#random_forest_cv_score = cross_val_score(RandomForestClassifier(), X, y, cv = 5, n_jobs = -1)
#print "RandomForestClassifier"
#print "Mean accuracy:", numpy.mean(random_forest_cv_score)
#print "Mean standard deviation:", numpy.std(random_forest_cv_score)

##### fit SVM model
#svm_estimator = SVC(class_weight = 'balanced', kernel = 'linear')
#svm_cv_score = cross_val_score(svm_estimator, X, y, cv = 5, n_jobs = -1)
#print "SVM"
#print "Mean accuracy:", numpy.mean(svm_cv_score)
#print "Mean standard deviation:", numpy.std(svm_cv_score)

##### fit GaussianNB model
#gaussiannb_estimator = GaussianNB()
#gaussiannb_cv_score = cross_val_score(gaussiannb_estimator, X, y, cv = 5, n_jobs = -1)
#print "GaussianNB"
#print "Mean accuracy:", numpy.mean(gaussiannb_cv_score)
#print "Mean standard deviation:", numpy.std(gaussiannb_cv_score)

##### train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

##### logistic regression
start = time.time()

logistic_estimator.fit(X_train, y_train)
y_pred = logistic_estimator.predict(X_test)

end = time.time()

print "classification_report for logisitic regression"
print classification_report(y_test, y_pred)
print "Time taken: ", end - start

##### random forest
#start = time.time()

#random_forest_estimator.fit(X_train, y_train)
#y_pred = random_forest_estimator.predict(X_test)

#end = time.time()

#print "classification_report for random forest"
#print classification_report(y_test, y_pred)
#print "Time taken: ", end - start

##### SVM
#start = time.time()

#svm_estimator.fit(X_train, y_train)
#y_pred = svm_estimator.predict(X_test)

#end = time.time()

#print "classification_report for SVM"
#print classification_report(y_test, y_pred)
#print "Time taken: ", end - start

##### GaussianNB
#start = time.time()

#gaussiannb_estimator.fit(X_train, y_train)
#y_pred = gaussiannb_estimator.predict(X_test)

#end = time.time()

#print "classification_report for gaussiannb_estimator"
#print classification_report(y_test, y_pred)
#print "Time taken: ", end - start

X_scaled = scale(X)

##### train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.33, random_state = 42)

##### logistic regression
start = time.time()

logistic_estimator.fit(X_train, y_train)
y_pred = logistic_estimator.predict(X_test)

end = time.time()

print "classification_report for scaled logisitic regression without grid"
print classification_report(y_test, y_pred)
print "Time taken: ", end - start

####################################################################################################
####################################################################################################
#####

logistic_regression_estimator = LogisticRegressionCV(Cs = 100, scoring = 'accuracy', cv = 5, class_weight = 'balanced', n_jobs = -1)
logistic_regression_estimator.fit(X_train, y_train)
y_pred = logistic_regression_estimator.predict(X_test)

print "classification_report for scaled logisitic regression with grid"
print classification_report(y_test, y_pred)

####################################################################################################
####################################################################################################
##### plot one, the best predictor against the output

print logistic_regression_estimator.C

index_of_best_predictor = numpy.argmax(numpy.abs(logistic_regression_estimator.coef_))
coefficient_of_best_predictor = logistic_regression_estimator.coef_[index_of_best_predictor]
intercept = logistic_regression_estimator.intercept_

best_predictor = X_scaled[:, index_of_best_predictor]

http://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression
http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html

https://pypi.python.org/pypi/scikit-neuralnetwork

http://www.robots.ox.ac.uk/~az/lectures/ml/2011/lect4.pdf




