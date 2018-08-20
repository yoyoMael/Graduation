#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount:
            labelCount[currentLabel] = 1
        else:
            labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = labelCount[key]/numEntries
        shannonEnt += -prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [x[i] for x in dataSet]
        uniqueValue = set(featList)
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet, i, value)

from sklearn.model_selection import  train_test_split
train_test_split(a,b, stratify)





