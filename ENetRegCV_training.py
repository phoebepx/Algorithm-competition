# coding:utf-8
# created by Phoebe_px on 2017/5/25
'''
ElasticNet回归构建二分类器，对sample中不同比例正负样本training,提取特征
'''

from math import sqrt, fabs, exp
import matplotlib.pyplot as plot
from sklearn.linear_model import enet_path
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
import pickle
def normalize(data):
    X = data.iloc[:,2:-1].replace(numpy.inf, numpy.nan).fillna(0).values
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
    X=numpy.array(xNormalized)
    return X,nrow,ncol

def reg_cv(data):
    X,nrow,ncol=normalize(data)
    label =list(data.iloc[:,-1])
    meanLabel = sum(label)/nrow
    sdLabel = sqrt(sum([(label[i] - meanLabel) * (label[i] - meanLabel) for i in range(nrow)])/nrow)

    labelNormalized = [(label[i] - meanLabel)/sdLabel for i in range(nrow)]
    Y = numpy.array(labelNormalized)

    alpha = 1.0
    #number of cross validation folds
    nxval = 10
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

    alphas, coefs, _ = enet_path(X, Y,l1_ratio=0.8, fit_intercept=False, return_models=False)

    nattr, nalpha = coefs.shape

    #find coefficient ordering
    nzList = []
    for iAlpha in range(1,nalpha):
        coefList = list(coefs[: ,iAlpha])
        nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
        for q in nzCoef:
            if not(q in nzList):
                nzList.append(q)

    #make up names for columns of X
    names = ['V' + str(i) for i in range(ncol)]
    nameList = [names[nzList[i]] for i in range(len(nzList))]

    alphaStar = plotAlphas[idxMax]
    indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
    indexStar = max(indexLTalphaStar)

    #here's the set of coefficients to deploy
    coefStar = list(coefs[:,indexStar])

    absCoef = [abs(a) for a in coefStar]

    #sort by magnitude
    coefSorted = sorted(absCoef, reverse=True)

    idxCoefSize = [absCoef.index(a) for a in coefSorted if not(a == 0.0)]

    namesList2 = [names[idxCoefSize[i]] for i in range(len(idxCoefSize))]

    return coefStar,namesList2
sample_path='./sample/'
with open('result_f.txt','w') as f:
    for i in range(1,11):
        fi='down_'+str(i)+'.pkl'

        coef,namelist2=reg_cv(pickle.load(open(sample_path+fi,'rb')))
        coef=[str(i) for i in coef]
        f.write(fi+'\n')
        f.write('\t'.join(coef))
        f.write('\n')
        f.write('\t'.join(namelist2))
        f.write('\n\n')




