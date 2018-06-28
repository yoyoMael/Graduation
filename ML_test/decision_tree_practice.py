#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

from math import math
import operator
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):
  numEntries = len(dataSet)
  labelCounts = {} #use {} to initiate a dictionary
  for featVec in dataSet:
    currentLabel = featVec[-1]
    if currentLabel not in labelCounts.keys():
      labelCounts[currentLabel] = 0
      labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
      prob = float(labelCounts[key])/numEntries
      shannonEnt += -prob*log(prob, 2)
    return shannonEnt # We can also use Gini impurity

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
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = 0.0
    for value in uniqueVals:
      subDataSet = splitDataSet(dataSet, i, value)
      prob = len(subDataSet)/float(len(dataSet))
      newEntropy += prob * calcShannonEnt(subDataSet)
    infoGain = baseEntropy - newEntropy
    if infoGain > bestInfoGain:
      bestInfoGain = infoGain
      bestFeature = i
  return bestFeature

def majorityCnt(classList):
  classCount = {}
  for vote in classList:
    if vote not in classCount.keys():
      classCount[vote] = 0
    classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

def createTree(dataSet, labels):
  classList = [example[-1] for example in dataSet]
  if classList.count(classList[0] == len(classList)):
    return classList[0] # If datas are of the same class, stop
  if len(dataSet[0]) == 1:
    return majorityCnt(classList) # If there is only 1 feature left, return the class
  bestFeat = chooseBestFeatureToSplit(dataSet)
  bestFeatLabel = labels[bestFeat]
  myTree = {bestFeatLabel:{}}
  del(labels[bestFeat])
  featValues = [example[bestFeat] for example in dataSet]
  uniqueVals = set(featValues)
  for value in uniqueVals:
    subLabels = labels[:]
    myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
  return myTree

def classify(inputTree, featLabels, testVec):
  firstStr = inputTree.keys()[0]
  secondDict = inputTree[firstStr]
  featIndex = featLabels.index(firstStr)
  for key in secondDict.keys():
    if testVec[featIndex] == key:
      if type(secondDict[key]).__name__ =='dict':
        classLabel = classify(secondDict[key], featLabels,testVec)
      else:
        classLabel = secondDict[key]
  return classLabel




