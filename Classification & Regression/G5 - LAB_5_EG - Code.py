# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:54:58 2020

@authors: andré luís & josé castro
"""
# ==============================================
# -----------Libraries
# ==============================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import keras as k
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
#from keras import optimizers
import tensorflow as tf
#import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn import linear_model


# =============================================================================
# Classification
# =============================================================================

#import data
xtrain = np.load("Cancer_Xtrain.npy")
ytrain = np.ravel(np.load("Cancer_ytrain.npy"))

xtest = np.load("Cancer_Xtest.npy")
ytest = np.load("Cancer_ytest.npy") 

# =============================================================================
# Support Vector Machines
# =============================================================================

#find the optimal parameters using grid search
possible_params = [
    { 'C': [0.1, 1, 10, 100, 1000, 10000],
     'gamma': ['scale', 10, 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf', 'linear']
     },
    ]

opt_params = GridSearchCV(
    SVC(),
    possible_params,
    cv=5, 
    scoring='accuracy',
    verbose=0)

opt_params.fit(xtrain, ytrain)
print(opt_params.best_params_)

classifier = SVC(kernel='linear', C=100)
classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)

accuracy = accuracy_score(ytest, ypred)
print("\nAccuracy Of SVM for Cancer prediction: ", accuracy)
print(classification_report(ytest, ypred))

plot_confusion_matrix(classifier, xtest, ytest, values_format = "d", 
                      display_labels = ["Cancer", "No Cancer"])

# =============================================================================
# Decision Trees
# =============================================================================

xtraindt, xvalidation, ytraindt, yvalidation = train_test_split(xtrain, ytrain, test_size = 0.2)

dt_classifier = DecisionTreeClassifier()
dtree = dt_classifier.fit(xtraindt, ytraindt)

plot_tree(dtree,
          filled = True, 
          rounded = True,
          class_names = ['Cancer', 'No Cancer'],
          feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x9', 'x10'])

plot_confusion_matrix(dt_classifier, xvalidation, yvalidation, 
                      display_labels = ["Cancer", "No Cancer"])

#build a pruned tree for each value of alpha
path = dtree.cost_complexity_pruning_path(xtraindt, ytraindt)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(ccp_alpha = ccp_alpha)
    clf_dt.fit(xtraindt, ytraindt)
    clf_dts.append(clf_dt)

#find optimum value for alpha
train_scores = [clf_dt.score(xtraindt, ytraindt) for clf_dt in clf_dts]
validation_scores = [clf_dt.score(xvalidation, yvalidation) for clf_dt in clf_dts]

fig , ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for Training and Validation sets')
ax.plot(ccp_alphas, train_scores, marker='x', label='train')
ax.plot(ccp_alphas, validation_scores, marker='x', label='validation')
ax.legend()
ax.show()

#build final tree
dt_classifier_pruned = DecisionTreeClassifier(ccp_alpha = 0.0195)
dtree_pruned = dt_classifier_pruned.fit(xtrain, ytrain)

plot_tree(dtree_pruned,
          filled = True, 
          rounded = True,
          class_names = ['Cancer', 'No Cancer'],
          feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x9', 'x10'])

ypred = dt_classifier_pruned.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("\nAccuracy Of Decision Trees for Cancer prediction: ", accuracy)
print(classification_report(ytest, ypred))
plot_confusion_matrix(dt_classifier_pruned, xtest, ytest, 
                      display_labels = ["Cancer", "No Cancer"])

# =============================================================================
# Regression
# =============================================================================

# =============================================================================
# Manipulating the data
# =============================================================================
# load data from files
X_train = np.load('Real_Estate_Xtrain.npy')  #(404,13)
y_train = np.load('Real_Estate_ytrain.npy')  #(404)

X_test = np.load('Real_Estate_Xtest.npy')    #(102,13)
y_test = np.load('Real_Estate_ytest.npy')    #(102)


# get sizes of data and labels, for all dimensions
#X_train_size = X_train.shape   #(404,13)
#y_train_size = Y_train.shape   #(404)
#X_test_size = X_test.shape     #(102,13)
#y_test_size = Y_test.shape    #(102)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Check for outliers
#from sklearn.ensemble import IsolationForest
#outliers = IsolationForest(random_state=0, contamination="auto").fit_predict(X_train)
#outliers = np.squeeze(np.argwhere(outliers==-1))

 # Convert the dictionary into DataFrame  
#X_train = pd.DataFrame(X_train)    
#y_train = pd.DataFrame(y_train)     
#X_train = X_train.drop(X_train.index[outliers])
#y_train = y_train.drop(y_train.index[outliers])

# split the training in subsets, 30% for validation and rest for training

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

# =============================================================================
# MLP with Early Stopping
# =============================================================================

# create sequential model and add layers
mlp_model = k.Sequential()
mlp_model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
mlp_model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#mlp_model.add(Dense(8, kernel_initializer='normal', activation='relu'))
mlp_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
mlp_model.add(Dense(units = 1))
    #1.2.4 - Summary of Neural Network:
mlp_model.summary()


    #1.2.5 - Early stopping monitor:
callbacks = tf.keras.callbacks.EarlyStopping(patience=10, verbose=0, mode='auto',baseline=None, restore_best_weights=True)

    #1.2.6 Compile and fit the MLP to your training and validation data:
        # Compile:

           
#mlp_model.compile(optimizer='adam',loss='mean_squared_error')
mlp_model.compile(optimizer='adam',loss='mean_absolute_error')


        # Fit:

history = mlp_model.fit(x=X_train,y=y_train, batch_size=12, epochs=200,validation_data=(X_validation, y_validation),callbacks=[callbacks], verbose=0)
#history = mlp_model.fit(x=X_train,y=y_train, batch_size=50, epochs=200,validation_data=(X_validation, y_validation),callbacks=[callbacks], verbose=0)

    #1.2.7 Plot the evolution of the training loss and validation loss:
            
            #Plot training Loss:
plt.figure()
plt.plot(history.history['loss'])

            #Plot validation loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'])


    #1.2.8 Evaluate performance on the test data:
y_predict = mlp_model.predict(X_test, batch_size=200)

#Testar a accuracy:
# evaluate the model
#train_mse = mlp_model.evaluate(X_train, y_train, verbose=0)
#test_mse = mlp_model.evaluate(X_test, y_test, verbose=0)

#print(test_mse)


mse_mlp = mean_squared_error(y_test,y_predict)
print(mse_mlp)

absolute_mlp = mean_absolute_error(y_test, y_predict)
print(absolute_mlp)

max_error_mlp = max_error(y_test, y_predict)
print(max_error_mlp)

# =============================================================================
# #Lasso
# =============================================================================

#Fit the model
linreg = linear_model.Lasso(alpha=0.5)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)                            

#Testar a accuracy:
#print(linreg.score(X_test,y_test))
mse_Lasso = mean_squared_error(y_test,y_pred)
print(mse_Lasso)

absolute_Lasso = mean_absolute_error(y_test, y_pred)
print(absolute_Lasso)

max_error_Lasso = max_error(y_test, y_pred)
print(max_error_Lasso)












