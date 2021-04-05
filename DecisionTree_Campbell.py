#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class DecisionTree:
    def __init__(self, X, y, max_depth= 2, min_leaf_size= 1,depth= 0,classes= None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.N = len(y)
        self.depth = depth
        
        if classes is None:
            self.classes = np.unique(self.y)
        else:
            self.classes = classes
        
        self.class_count = []
        for z in self.classes:
            self.class_count.append(np.sum(z==self.y))
            
        self.class_count = np.array(self.class_count)
        self.prediction = self.classes[np.argmax(self.class_count)]
        class_ratios = self.class_count / self.N
        self.gini = 1 - np.sum(class_ratios **2)
        
        if (depth == max_depth) or ( self.gini == 0):
            self.axis = None
            self.t = None
            self.left = None
            self.right = None
            return
        
        self.axis = np.random.choice(range(self.X.shape[1]))
        self.t = 0
        best_gini = 2
        
        for obs in range(self.X.shape[1]):
            col_values = self.X[:, obs].copy()
            col_values = np.sort(col_values)
                
            for k in range (len(col_values)):
                sel = self.X[:,obs] <= col_values[k]
                n_left = np.sum(sel)
                n_right = np.sum(~sel)
                
                if (n_left >= min_leaf_size) & (n_right >= min_leaf_size):
                    _,left_counts = np.unique(self.y[sel], return_counts=True)
                    class_ratios = left_counts / n_left
                    left_gini = 1 - np.sum((class_ratios **2))
                    
                    _,right_counts = np.unique(self.y[~sel], return_counts=True)
                    class_ratios = right_counts / n_right
                    right_gini = 1-np.sum((class_ratios **2))
                    gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)
                    
                    if (gini <= best_gini):
                        best_gini = gini
                        self.axis = obs
                        if((k +1)== len(col_values)):
                            self.t = col_values[k]
                        else:
                            self.t = (col_values[k]+col_values[k+1])/2
        sel = self.X[:,self.axis]<= self.t
        if  (np.sum(sel) < min_leaf_size) or (np.sum(~sel) < min_leaf_size) or (best_gini == 2):
            self.axis = None
            self.t = None
            self.left   = None
            self.right = None
            return

        
        self.left = DecisionTree(self.X[sel,:], self.y[sel], max_depth, min_leaf_size,depth+1, self.classes)
        self.right = DecisionTree(self.X[~sel,:], self.y[~sel], max_depth, min_leaf_size,depth+1, self.classes)
         
                
    def classify_row(self,row):
        row = np.array(row)
        
        if self.left == None or self.right == None:
            return self.prediction 
    
        if row[self.axis] <= self.t:
            return self.left.classify_row(row)
        else:
            return self.right.classify_row(row)           
        
    def predict(self,X):
        X = np.array(X)
        predictions = []
        
        for num in range(X.shape[0]):
            row = X[num,:]
            predictions.append(self.classify_row(row))
            
        predictions = np.array(predictions)
        return predictions
    
    
    def score(self,X,y):
        X = np.array(X)
        y = np.array(y)
        N = len(y)
        y_prediction = self.predict(X)     
        accuracy = (np.sum(y== y_prediction))/ N   
        return accuracy
        
        
    def print_tree(self):
        msg = '  ' * self.depth + '* '
        msg += 'Size: ' + str(self.N) + ' '
        msg += str(list(self.class_count))
        msg += ', Gini: ' + str(round(self.gini,2))           
        if(self.left != None):
            msg += ', Axis: ' + str(self.axis)
            msg += ', Cut: ' + str(round(self.t,2))
        else:
            msg += ', Predicted Class: ' + str(self.prediction)
                
        
        print(msg)
        
        if self.left != None:
            self.left.print_tree()
            self.right.print_tree()