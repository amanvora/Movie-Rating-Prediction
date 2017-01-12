#####################################################
# Date      :       12th Dec 2016                   #
# Name      :       Aman Vora                       #
# USC ID    :       4057796804                      #
# Email     :       amanvora@usc.edu                #
# Written for EE660 final project                   #
#						    #
# Please ensure you've ran preproc.py and generated #
# 'preprocessed.txt' and 'imdbScores.txt'           #
# This script retrains the model with all the	    #
# training data and finds the in-sample and 	    #
# out-of-sample error. It plots these against the   #
# training size.				    #
#####################################################
# Import dependencies and/or other modules
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load workspace variable from saved file
D_Arr = np.loadtxt('preprocessed.txt')
imdbScores = np.loadtxt('imdbScores.txt')
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
X = D_Arr[:,:-1]
y = D_Arr[:,-1]
trainSet = 0.1*np.array(range(1,10))
inSample = []
outOfSample = []
for i in trainSet:
    print ('Training size: %.2f'%i)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=1-i, random_state=0)
    # Selected model
    randFor = RandomForestRegressor(max_depth=15, random_state=0, n_estimators=35, max_features=0.5)
    
    # Fit the model to training data
    randFor.fit(X_train,y_train)
    inS = np.mean((randFor.predict(X_train) - y_train) ** 2)
    # Obtain in-sample error
    inSample.append(inS)
    print("Random Forest: In sample mean square error: %.4f" % inS)
    # Compute the predicted output
    yHat = randFor.predict(X_test)
    # Obtain out-of-sample error
    outS = np.mean((yHat - y_test) ** 2)
    outOfSample.append(outS)
    print("Random Forest: Out-of sample mean square error: %.4f" % outS)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f\n' % randFor.score(X_test, y_test))

# Compute how many test samples are predicted 'close' to original score
yMean = np.mean(imdbScores)
yStd = np.std(imdbScores)

# Rescale the predicted output back to the original scale of the ground turth
movieScorePred = yHat*yStd+yMean
errorMargin = 0.1*np.array(range(1,16,2))
pred = []
for e in errorMargin:
    count = 0
    for i in range(len(y_test)):
        movieScore = y_test[i]*yStd+yMean
        if(abs(movieScorePred[i]-movieScore) < e):
            count += 1
    pred.append(count*100.0/len(y_test))
    print ('Prediction accuracy = %.2f with an error of %.1f allowed'% (count*100.0/len(y_test),e))
    
# Plot of Learning curve
figLC = plt
figLC.plot(trainSet,outOfSample,'g-',linewidth=2,label='Out-of Sample Error')
figLC.plot(trainSet,inSample,'b-',linewidth=2,label='In Sample Error')
figLC.grid()
figLC.title('Learning curve')
figLC.xlabel('Fraction of dataset for training')
figLC.ylabel('Mean Square Error')
figLC.legend(loc='upper right')
figLC.show()

# Plot of prediction accuracy against error tolerance
figAcc = plt
figAcc.plot(errorMargin,pred,'bo-')
figAcc.grid()
figAcc.title('Prediction accuracy of score given error margin')
figAcc.xlabel('Error margin tolerance')
figAcc.ylabel('Prediction accuracy in test set')
figAcc.show()
