#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

def createC1(dataSet):
  C1 = []
  for transaction in dataSet:
    for item in transaction:
      if not [item] in C1:
        C1.append([item])
  C1.sort()
  return map(frozenset, C1)

def scanD(D, Ck, minSupport):
  ssCnt = {}
  for tid in D:
    for can in Ck:
      if can.issubset(tid):
        if not ssCnt.has_key(can): ssCnt[can] = 1
      else: ssCnt[can] += 1
  numItems = float(len(D))
  retList = []
  supportData = {}
  for key in ssCnt:
    support = ssCnt[key]/numItems
    if support >= minSupport:
      reList.insert(0,key)
    supportData[key] = support
  return retList, supportData


