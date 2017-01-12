#####################################################
# Date      :       12th Dec 2016                   #
# Name      :       Aman Vora                       #
# USC ID    :       4057796804                      #
# Email     :       amanvora@usc.edu                #
# Written for EE660 final project                   #
#						    #
# Please ensure you've ran preproc.py and generated #
# 'preprocessed.txt' and 'imdbScores.txt' first     #
# This script performs a 5-fold cross validation on #
# each model over a range of its hyperparameters    #
#####################################################

# Constant vaiables
COL = 1
# Import dependencies and/or other modules
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Load workspace variable from saved file
D_Arr = np.loadtxt('preprocessed.txt')

# Import scikit-learn modules
from sklearn import linear_model, svm, cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

#-------------------Uncomment the line below to plot the covariance----------------#
#-------------------------matrix of the numerical features-------------------------#
##import matplotlib.pyplot as plt
##cax = plt.matshow(np.cov(D_Arr[:,:-1].T))
##cax = plt.matshow(np.cov(D_Arr[:,0:15].T))
##plt.clim(-1,1)
##plt.colorbar(cax)
##plt.title('Covariance matrix of numerical features')
##plt.show()
#----------------------------------------------------------------------------------#
# Input pf the dataset
X = D_Arr[:,:-1]

# Ground truth of the dataset
y = D_Arr[:,-1]

# Split the dataset in the ratio train:test = 0.9:0.1
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

# Create OLS linear regression object
regrOLS = linear_model.LinearRegression()

# Perform 5 fold cross-validation and store the MSE resulted from each fold
scores = cross_validation.cross_val_score(regrOLS,X_train,y_train,scoring='mean_squared_error',cv=5)

# Note: Due to a known issue in scikit-learn the results return are flipped in sign
print ('OLS: Least CV error: %.2f\n' % np.min(-scores))
#------------------------------------------------------------------------------#
#---------------- Cross validation for Ridge and Lasso ------------------------#
#------------------------------------------------------------------------------#
# Range of hyperparamters to choose for CV
lambdas = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 1, 10]
for l in lambdas:
    print ('Lambda = %.5f'% l)
    # Start time for the 5-fold CV
    start = time.time()
    # Create ridge regression object
    regrRidge = linear_model.Ridge(alpha=l)
    scores = cross_validation.cross_val_score(regrRidge,X_train,y_train,scoring='mean_squared_error',cv=5)
    end = time.time()
    t = end-start
    print ('Ridge: Least CV error: %.2f and time : %.3f' % (np.min(-scores), t))
    start = time.time()
    # Create lasso object
    regrLasso = linear_model.Lasso(alpha=l)
    scores = cross_validation.cross_val_score(regrLasso,X_train,y_train,scoring='mean_squared_error',cv=5)
    # Measure and compute time for the 5-fold CV
    end = time.time()
    t = end-start
    print ('Lasso: Least CV error: %.2f and time : %.3f' % (np.min(-scores), t))
    print ('\n')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-------------------- Cross validation for Elastic Net ------------------------#
#------------------------------------------------------------------------------#
# Range of hyperparamters to choose for CV
l1Ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
for l in lambdas:
    print ('Lambda = %.5f'% l)
    for l1R in l1Ratios:
        start = time.time()
        # Create elastic net object
        regrElasNet = linear_model.ElasticNet(alpha=l, l1_ratio=l1R)
        scores = cross_validation.cross_val_score(regrElasNet,X_train,y_train,scoring='mean_squared_error',cv=5)
        end = time.time()
        t = end-start
        print ('Elastic Net: l1Ratio = %.2f, Least CV error: %.2f and time : %.3f' % (l1R, np.min(-scores), t))
    print ('\n')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#------------ Cross validation for Support Vector Regressor -------------------#
#------------------------------------------------------------------------------#
# Range of hyperparamters to choose for CV
C = [0.01,0.1,1,10,20,50]
eps = [0.0005,0.001,0.01,0.05,0.1,0.5,1,10,100]
for c in C:
    print ('C = %.5f'% c)
    for e in eps:
        start = time.time()
        # Create SVR object
        svr = svm.SVR(C = c, epsilon = e)
        scores = cross_validation.cross_val_score(svr,X_train,y_train,scoring='mean_squared_error',cv=5)
        end = time.time()
        t = end-start
        print ('SVR: eps = %.4f, Least CV error: %.2f and time : %.3f'% (e, np.min(-scores), t))
    print ('\n')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#------------- Cross validation for Random Forest Regressor -------------------#
#------------------------------------------------------------------------------#

# Range of hyperparamters to choose for CV
numTrees = [1, 2, 5, 10, 20, 35, 50, 100, 200]
maxFeatures = [0.25, 0.5, 0.75, 1]
maxDepth = [3, 6, 8, 10, 15, 25]
for n in numTrees:
    for mf in maxFeatures:
        for d in maxDepth:
            print ('Number of trees/estimators = %d, max depth = %d'% (n,d))
            start = time.time()
            # Create Random Forest Regressor object
            randFor = RandomForestRegressor(max_depth=d, random_state=0, n_estimators=n, max_features=mf)
            scores = cross_validation.cross_val_score(randFor,X_train,y_train,scoring='mean_squared_error',cv=5)
            end = time.time()
            t = end-start
            print ('Random forest regressor: %% of features = %.2f, Least CV error: %.2f and time : %.3f'% (100*mf, np.min(-scores), t))
        print ('\n')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#------------- Cross validation for regressor using Boosting  -----------------#
#------------------------------------------------------------------------------#

# Range of hyperparamters to choose for CV
numTrees = [1, 2, 5, 10, 20, 35, 50, 100, 200]
lossType = ['linear', 'square', 'exponential']
maxDepth = [3, 6, 8, 10, 15, 25]
for l in lossType:
    for n in numTrees:
        for d in maxDepth:
            print ('Number of trees/estimators = %d, max depth = %d'% (n,d))
            start = time.time()
            # Create Boosting Regressor object
            boosting = AdaBoostRegressor(DecisionTreeRegressor(max_depth=d),random_state=0, n_estimators=n, loss=l)
            scores = cross_validation.cross_val_score(boosting,X_train,y_train,scoring='mean_squared_error',cv=5, n_jobs=1)
            end = time.time()
            t = end-start
            print ('Regressor using boosting: loss type = %s, Least CV error: %.2f and time : %.3f'% (l, np.min(-scores), t))
        print ('\n')
#------------------------------------------------------------------------------#
