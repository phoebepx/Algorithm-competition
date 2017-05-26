# coding:utf-8
# created by Phoebe_px on 2017/5/23
'''
ENetRegCV_training后选择的特征列，基于trainingset_2016-03-10_2016-04-11_2016-04-11_2016-04-16.pkl训练的最优参数，预测04-16——04-20高潜用户购买
'''
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.linear_model import enet_path
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
import pickle
from math import sqrt, fabs, exp
from clean import get_trainingset
from clean import get_testset
import pandas as pd
#read data
column = [62, 107, 70, 78, 73, 71, 64, 51, 63, 77, 89, 18, 49, 66, 69, 52, 76, 108, 81, 75, 5, 61, 79, 95, 74, 3, 84]
data=pickle.load(open('./pickle_data/trainingset_2016-03-10_2016-04-11_2016-04-11_2016-04-16.pkl','rb'))
X = data.iloc[:,2:-1].replace(numpy.inf, numpy.nan).fillna(0).iloc[:,column].values
nrow,ncol = X.shape[0],X.shape[1]
xMeans = []
xSD = []
for i in range(ncol):
    col = [X[j][i] for j in range(nrow)]
    mean = sum(col)/nrow
    xMeans.append(mean)
    colDiff = [(X[j][i] - mean) for j in range(nrow)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrow)])
    stdDev = sqrt(sumSq/nrow)
    xSD.append(stdDev)

#use calculate mean and standard deviation to normalize xNum
xNormalized = []
for i in range(nrow):
    rowNormalized = []
    for j in range(ncol):
        if xSD[j]!=0:
            rowNormalized.append((X[i][j] - xMeans[j])/xSD[j])
        else:
            rowNormalized.append(0.)
    xNormalized.append(rowNormalized)
label =list(data.iloc[:,-1])
meanLabel = sum(label)/nrow
sdLabel = sqrt(sum([(label[i] - meanLabel) * (label[i] - meanLabel) for i in range(nrow)])/nrow)

labelNormalized = [(label[i] - meanLabel)/sdLabel for i in range(nrow)]
Y = numpy.array(labelNormalized)
X=numpy.array(xNormalized)



alpha = 1.0
#number of cross validation folds
nxval = 10
print(numpy.isnan(X).any())

for ixval in range(nxval):
    #Define test and training index sets
    idxTest = [a for a in range(nrow) if a%nxval == ixval]
    idxTrain = [a for a in range(nrow) if a%nxval != ixval]

    #Define test and training attribute and label sets
    xTrain = numpy.array([X[r] for r in idxTrain])
    xTest = numpy.array([X[r] for r in idxTest])
    labelTrain = numpy.array([Y[r] for r in idxTrain])
    labelTest = numpy.array([Y[r] for r in idxTest])
    alphas, coefs, _ = enet_path(xTrain, labelTrain,l1_ratio=0.8, fit_intercept=False, return_models=False)
    #apply coefs to test data to produce predictions and accumulate
    if ixval == 0:
        pred = numpy.dot(xTest, coefs)
        yOut = labelTest
    else:
        #accumulate predictions
        yTemp = numpy.array(yOut)
        yOut = numpy.concatenate((yTemp, labelTest), axis=0)

        #accumulate predictions
        predTemp = numpy.array(pred)
        pred = numpy.concatenate((predTemp, numpy.dot(xTest, coefs)), axis = 0)


#calculate miss classification error
misClassRate = []
_,nPred = pred.shape
for iPred in range(1, nPred):
    predList = list(pred[:, iPred])
    errCnt = 0.0
    for irow in range(nrow):
        if (predList[irow] < 0.0) and (yOut[irow] >= 0.0):
            errCnt += 1.0
        elif (predList[irow] >= 0.0) and (yOut[irow] < 0.0):
            errCnt += 1.0
    misClassRate.append(errCnt/nrow)

#find minimum point for plot and for print
minError = min(misClassRate)
idxMin = misClassRate.index(minError)
plotAlphas = list(alphas[1:len(alphas)])
#calculate AUC.
idxPos = [i for i in range(nrow) if yOut[i] > 0.0]
yOutBin = [0] * nrow
for i in idxPos: yOutBin[i] = 1
auc = []
for iPred in range(1, nPred):
    predList = list(pred[:, iPred])
    aucCalc = roc_auc_score(yOutBin, predList)
    auc.append(aucCalc)

maxAUC = max(auc)
idxMax = auc.index(maxAUC)

alphaStar = plotAlphas[idxMax]
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)
coefStar = coefs[:,indexStar]
sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'
sub_user_index, predf = get_testset(sub_start_date, sub_end_date,)
predf=predf.replace(numpy.inf, numpy.nan).fillna(0).iloc[:,column].values
sub_pred = pd.DataFrame(numpy.dot(predf,coefStar))
result=pd.concat([sub_user_index,sub_pred],axis=1)
result.to_csv('0523.csv',index=False, index_label=False)


