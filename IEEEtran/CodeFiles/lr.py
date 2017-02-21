from __future__ import division
import json
import numpy as np
import argparse
import time
import math
import os.path
from sklearn import datasets
from sklearn import linear_model
import polynest as pn

def hinge_loss(w,x,y,compute_lr):
    """ evaluates hinge loss and its gradient at w

    rows of x are data points
    y is a vector of labels
    """
    loss,grad = 0,0
    #print "Inside hinge:"
    #print "y: %s" %y
    i = 0
    for (x_,y_,compute_lr_) in zip(x,y,compute_lr):
        #v = y_*np.dot(w,x_)
        v = y_*compute_lr_
        loss += max(0,1-v)
	#grad += 0 if v > 1 else -y_*x_ 
        grad += 0 if v > 1 else -y_*x_*compute_lr_*(1.0-compute_lr_) #derivative of sigmoid(wx) is sigmoid(1-sigmoid) * x
        i += 1
    return (loss,grad)

'''
def grad_desc_hinge(x,y,w,step=.000001,thresh=0.000001):
    grad = np.inf
    ws = np.zeros((116,0))
    ws = np.hstack((ws,w.reshape(116,1)))
    step_num = 1
    delta = np.inf
    loss0 = np.inf
    while np.abs(delta)>thresh:
        loss,grad = hinge_loss(w,x,y)
        delta = loss0-loss
        loss0 = loss
        grad_dir = grad/np.linalg.norm(grad)
        w = w-step*grad_dir/step_num
        ws = np.hstack((ws,w.reshape((116,1))))
        step_num += 1
    return np.sum(ws,1)/np.size(ws,1), None
'''

def logistic_func(theta, x):
    denom = x.dot(theta)
    #print "x is %s" %x
    #print "theta is %s" %theta
    #print "denom is %s" %denom
    lf = float(1) / (1 + math.e**(-x.dot(theta)))
    return lf

def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc

def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

def normalize(X):
    for col in range(len(X[0])):
        mean = 0
        for row in range(len(X)):
            mean += X[row][col]
        mean /= len(X)
        var = 0
        for row in range(len(X)):
            var += math.pow(X[row][col] - mean, 2)
        std = math.sqrt(var) / len(X)
        for row in range(len(X)):
            if std > 0:
                X[row][col] = (X[row][col] - mean) / std
    #print "X is %s " %X[0]
    #print "Finished printing X " 
    return X

#def grad_desc(theta_values, X, y, mode, cmp, lr=.0001, converge_change=.0001):
def grad_desc(theta_values, X, y, mode, cmp, lr=.000001, converge_change=.000001):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    cost_iter = []
    #X = normalize(X)
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = np.inf
    i = 1
    lambdaFac = 0
    mu = 1.0
    prev_velocity = 0
    while(np.abs(change_cost) > converge_change):
        if cmp == "nesterov":
            theta_values_param = theta_values + mu * prev_velocity #the difference from polyak
        else:
            theta_values_param = theta_values
        old_cost = cost
        compute_lr = logistic_func(theta_values_param, X)
        if mode == "loglh":
            grad = log_gradient(theta_values_param, X, y)
        elif mode == "hinge":
            (loss,grad) = hinge_loss(theta_values, X, y, compute_lr)
        if cmp == "l2reg":
            lambdaFac = 1
            theta_values = theta_values - (lr * grad) + (lambdaFac * theta_values) #lr stands for learning rate
        elif cmp == "polyak":
            (theta_values, prev_velocity) = pn.polyak(theta_values, lr, lambdaFac, grad, prev_velocity, mu)
        elif cmp == "nesterov":
            # polyak is called as the difference from nesterov is only in the weight vector passed to the gradient computation procedure
            (theta_values, prev_velocity) = pn.polyak(theta_values, lr, lambdaFac, grad, prev_velocity, mu)
        else:
            #print "Gradient used: %s" %grad
            theta_values = theta_values - (lr * grad) + (lambdaFac * theta_values) #lr stands for learning rate
        #print "lambdaFac is %s" % lambdaFac
        if  mode == "hinge":
            #cost = loss
            cost = cost_func(theta_values, X, y)
        else:
            cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        #print "change_cost in iteration %d is %s, old_cost: %s, cost: %s" %(i,change_cost,old_cost,cost)
        i+=1
    print "Num iterations: %s" %i
    return i,theta_values, np.array(cost_iter)

def pred_values(theta, X, mode, hard=True):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#    X = normalize(X)
    pred_prob = logistic_func(theta, X)
    if mode == "loglh":
        pred_value = np.where(pred_prob >= .5, 1, 0)
    elif mode == "hinge":
        pred_value = np.where(pred_prob >= .5, 1, -1)
    if hard:
        return pred_value
    return pred_prob

def load_dataset(mode):
    data = datasets.load_iris()
    X = data.data[:100, :2]
    y = data.target[:100]
    X_full = data.data[:100, :]
    print X
    if mode == "hinge":
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
    print y
    return (data, X, X_full, y)

def flip(y,mode):
    if mode == "loglh":
        return np.logical_not(y)
    elif mode == "hinge":
        for i in range(len(y)):
            assert y[i] == 1 or y[i] == -1, "class label for hinge loss can be 1 or -1"
            if y[i] == 1:
                y[i] = -1
            elif y[i] == -1:
                y[i] = 1
        return y 

def train(X, y, mode, cmp):
    shape = X.shape[1]
    y_flip = flip(y,mode) #flip Setosa to be 1 and Versicolor to zero to be consistent
    betas = np.zeros(shape)
    iter,fitted_values, cost_iter = grad_desc(betas, X, y_flip, mode, cmp)
    print fitted_values
    return (iter,fitted_values, cost_iter, y_flip)

def test(fitted_values, X, mode):
    predicted_y = pred_values(fitted_values, X, mode)
    print "Predicted_y: "
    print predicted_y
    return predicted_y

def computeF1(predicted_y,y,mode):
    assert len(predicted_y) == len(y), "the length should match"
    targetNeg = 0
    if mode =="hinge":
        targetNeg = -1
    precision =0.0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if predicted_y[i] == y[i] and y[i] == 1: 
            TP += 1
        elif predicted_y[i] == y[i] and y[i] == targetNeg:
            TN += 1 
        elif predicted_y[i] != y[i] and y[i] == 1: 
            FP += 1
        elif predicted_y[i] != y[i] and y[i] == targetNeg: 
            FN += 1
    precision = 0
    recall = 0
    if TP == 0 and FP ==0:
        precision = 0
    else:
        precision = TP / (TP+FP)
    if TP ==0 and FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision == 0  or recall == 0:
        return (precision, recall, 0.0)
    else:
        return (precision, recall, 2*precision*recall/(precision + recall))

def buildFV(tids, X, y, matches, fv, attrs, mode, dir):
    #match = 0
    for tid in tids:
        match = matches[str(tid)]
        if mode == "hinge" and match == 0:
            match = -1
        tupleFv = [] 
        if "Amazon-GoogleProducts" in dir:
            for attrId in fv[str(tid)].keys():
                for dimId in fv[str(tid)][attrId].keys():
                    #if fv[str(tid)][attrId][dimId] == 0:
                    #    tupleFv.append(0.0000001)
                    #else:
                    tupleFv.append(fv[str(tid)][attrId][dimId]/1000.0) # converting 666 to 0.666 from the dbgen vector
        else:
            for attrId in fv[str(tid)].keys():
                tupleFv.append(fv[str(tid)][attrId]/1000.0) # converting 666 to 0.666 from the synthetic dataset
        X.append(tupleFv)
        y.append(match)
    return (np.array(X), np.array(y))

def run5FoldExp(args):
    dir = args.dataDir
    matchesFile = dir+"/matches.json"
    foldsFile = dir+"/5-folds.json"
    fvFile = dir+"/featurevector.json"
    attrFile = dir+"/attributes.json"
    with open(matchesFile,"r") as fm:
        matches = json.load(fm)
    with open(foldsFile,"r") as ff:
        folds = json.load(ff)
    with open(fvFile,"r") as ffv:
        fv = json.load(ffv)
    with open(attrFile,"r") as fa:
        attrs = len(json.load(fa))

    avgP = 0.0
    avgR = 0.0
    avgF1 = 0.0
    avgP_orig = 0.0
    avgR_orig = 0.0
    avgF1_orig = 0.0
    # retrieve all training and test in each fold.  
    avg_iter = 0
    for fid in range(len(folds)):
        X_train = []
        y_train = []
        X_test = []
        y_test = [] 
        trainIds = folds[fid][0]
        testIds = folds[fid][1]
        (X_train, y_train) = buildFV(trainIds, X_train, y_train, matches, fv, attrs, args.mode, dir) 
        (X_test, y_test) = buildFV(testIds, X_test, y_test, matches, fv, attrs, args.mode, dir) 
        (iter,fitted_values, cost_iter, y_flip) = train(X_train, y_train, args.mode, args.cmp)
        #test_y = y_flip
        test_y = flip(y_test,args.mode) #flip 
        predicted_y = test(fitted_values, X_test, args.mode)
        (precision, recall, F1)=computeF1(predicted_y,test_y, args.mode)
        print "Fold: %s, Precision: %s, Recall: %s, F1: %s" %(fid,precision,recall,F1)
        avgP += precision
        avgR += recall
        avgF1 += F1
        avg_iter += iter
        '''
        logreg = linear_model.LogisticRegression()
        logreg.fit(normalize(X_train), y_train)
        predicted_y = logreg.predict(normalize(X_test))
        (precision, recall, F1)=computeF1(predicted_y,test_y)
        print "Fold: %s, Precision_orig: %s, Recall_orig: %s, F1_orig: %s" %(fid,precision,recall,F1)
        avgP_orig += precision
        avgR_orig += recall
        avgF1_orig += F1
        '''
    avgP /= len(folds)
    avgR /= len(folds)
    avgF1 /= len(folds)
    avg_iter /= len(folds)
    print "Avg_iterations: %s, Mode: %s, avgP: %s, avgR: %s, avgF1: %s" %(avg_iter,args.mode,avgP,avgR,avgF1)
    '''
    avgP_orig /= len(folds)
    avgR_orig /= len(folds)
    avgF1_orig /= len(folds)  
    print "Mode: %s, avgP_orig: %s, avgR_orig: %s, avgF1_orig: %s" %(args.mode,avgP_orig,avgR_orig,avgF1_orig)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#   Assert(len(sys.argv) == 2, "Need 1 argument: filename")
#   filename = sys.argv[1]
#   parser.add_argument("-config",help="Config parameters file",type=str,required=True)
    parser.add_argument("-mode",help="loglh or hinge",type=str,required=True)
    parser.add_argument("-folds",help="True or False",type=str,required=True)
    parser.add_argument("-dataDir",help="path to the dataset",type=str,required=True)
    parser.add_argument("-cmp",help="l2reg or polyak or nesterov",type=str,required=True)
    args = parser.parse_args()

    '''
    test_y = y
    if args.mode == "loglh":
        (fitted_values, cost_iter, y_flip) = train(data, X, X_full, y)
        test_y = y_flip
    elif args.mode == "hinge":
        fitted_values = grad_descent(X,y,np.array((0,0)),0.001)
    predicted_y = test(fitted_values, X)
    '''
   
    if args.folds == "True":
        run5FoldExp(args)
    elif args.folds == "False":
        (data, X, X_full, y) = load_dataset(args.mode)
        (fitted_values, cost_iter, y_flip) = train(X, y, args.mode)
        test_y = y_flip
        predicted_y = test(fitted_values, X, args.mode)
        (precision, recall, F1)=computeF1(predicted_y,test_y)
        print "Precision: %s, Recall: %s, F1: %s" %(precision,recall,F1)
