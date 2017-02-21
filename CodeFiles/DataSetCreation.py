import json
from random import randint
from sets import Set
import math
import os
"""def dictPractice():
    i = 1
    row = list()
    dix = {}
    for i in range(1, 50, 1):
        row.append(randint(0,9))
    dix[i] = row

    i = 2
    row = list()
    for i in range(1, 50, 1):
        row.append(randint(0, 9))

    dix[i] = row

    print dix[i]


def practiceSet():
    setTest = Set()
    setTest.add(1)
    if not setTest.__contains__(1):
        print 'hi'
    else:
        print 'not hi'
"""

def creationFolder():
    if not os.path.exists('./LinearSeparable'):
        os.makedirs('./LinearSeparable')
    if not os.path.exists('./NonLinearSeparable'):
        os.makedirs('./NonLinearSeparable')

def linearSeparable(colNum, rowNum):
    attribute = list()
    featureDix = {}
    matchJson = {}
    for i in range(0 , rowNum/2 , 1):
        row = {}
        for j in range(0, colNum , 1):
            row[j] = float(randint(0, 500))
        featureDix[i] = row
        matchJson[i] = 1

    for i in range( rowNum/2, rowNum  , 1):
        row = {}
        for j in range(0, colNum , 1):
            row[j] = float(randint(501 , 1000))
        featureDix[i] = row
        matchJson[i] = 0

    for i in range(0, colNum , 1):
         attribute.append(i)
    """
        write to the files
    """
    try:
        with open('./LinearSeparable/featurevector.json','w') as outFile:
            json.dump(featureDix, outFile)
        with open('./LinearSeparable/matches.json','w') as outFile:
            json.dump(matchJson, outFile)

        with open('./LinearSeparable/attributes.json','w') as outFile:
            json.dump(attribute, outFile)
    except Exception:
        print Exception.args

def nonlinearSeparable(colNum, rowNum):
    attribute = list()
    featureDix = {}
    matchJson = {}
    for i in range(0 , rowNum , 1):
        row = {}
        for j in range(0, colNum , 1):
            row[j] = float(randint(0, 1000))
        featureDix[i] = row
        matchJson[i] = randint(0,1)

    jsonFeature = json.dumps(featureDix)
    jsonMatch = json.dumps(matchJson)
    for i in range(0, colNum , 1):
         attribute.append(i)

    # print jsonFeature
    # print jsonMatch
    # print json.dumps(attribute)
    """
        write to the files
    """
    try:
        with open('./NonLinearSeparable/featurevector.json','w') as outFile:
            json.dump(featureDix, outFile)
        with open('./NonLinearSeparable/matches.json','w') as outFile:
            json.dump(matchJson, outFile)
        with open('./NonLinearSeparable/attributes.json','w') as outFile:
            json.dump(attribute, outFile)
    except Exception:
        print Exception.args

def fiveFold(rowNum, k):

    fold = []
    validSet = list()
    for k in range(0, k , 1):
        setElements = Set()
        train = list()
        test = list()
        while len(setElements)<rowNum:
            element = randint(0, rowNum -1)
            if not setElements.__contains__(element):
                setElements.add(element)
                if train.__len__()< math.floor(0.80 * rowNum):
                    train.append(element)
                else:
                    test.append(element)

        validSet =  train , test
        # print validSet
        # fold = fold.append(validSet)
        fold.append(validSet)
    #print json.dumps(fold)
    try:
        with open('./NonLinearSeparable/5-folds.json','w') as outFile:
            json.dump(fold, outFile)
        with open('./LinearSeparable/5-folds.json','w') as outFile:
            json.dump(fold, outFile)
    except Exception:
        print Exception.args

def mainFunction(rowNum, colNum, k):
    creationFolder()
    linearSeparable(colNum, rowNum)
    nonlinearSeparable(colNum, rowNum)
    fiveFold(rowNum, k)

if __name__ == "__main__":

    mainFunction(20000, 50, 5) # parameters number of rows, number of  columns , number of folds































