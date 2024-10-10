import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
# However, importing the Python standard library is allowed: https://docs.python.org/3/library/


#from the Python standard library:
import csv

#code for evaluating the results:
""" 
#load the dataset from pt 1 and train and predict the tree
X, Y = Tree.load_dataset()
t = Tree.train(X,Y) 
Y_predict = Tree.predict(t,X) 

accuracy = sum(Y==Y_predict)/ len(Y) 
print('training accuracy:', accuracy)

# Calculate the counts of true positives, false positives, and false negatives
true_positives = sum((Y == 'Good') & (Y_predict == 'Good'))
false_positives = sum((Y == 'Bad') & (Y_predict == 'Good'))
false_negatives = sum((Y == 'Good') & (Y_predict == 'Bad'))

#precision: tp / (tp + fp)
precision = true_positives/(true_positives+ false_positives)
print('training precision:', precision)

#recall: tp / (tp + fn)
recall = true_positives/(true_positives + false_negatives)
print('training recall:', recall)


# train over half of the dataset
t = Tree.train(X[:,::2],Y[::2]) 
# test on the other half
Y_predict = Tree.predict(t,X[:,1::2]) 
accuracy = sum(Y[1::2]==Y_predict)/21. 
print( 'test accuracy:', accuracy)
# Calculate the counts of true positives, false positives, and false negatives
true_positives = sum((Y[1::2] == 'Good') & (Y_predict == 'Good'))
false_positives = sum((Y[1::2] == 'Bad') & (Y_predict == 'Good'))
false_negatives = sum((Y[1::2] == 'Good') & (Y_predict == 'Bad'))

#precision: tp / (tp + fp)
precision = true_positives/(true_positives+ false_positives)
print('test precision:', precision)

#recall: tp / (tp + fn)
recall = true_positives/(true_positives + false_negatives)
print('test recall:', recall)
"""
#results from evaluation:
""" training accuracy: 0.9285714285714286
training precision: 0.9545454545454546
training recall: 0.9130434782608695
test accuracy: 0.7142857142857143
test precision: 0.6923076923076923
test recall: 0.8181818181818182 """


#used these sources, mostly to help me with the calculations and working through the tree for inference:
"""
https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
https://www.geeksforgeeks.org/decision-tree/
https://www.w3schools.com/python/python_ml_decision_tree.asp 

"""
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.
'''

#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        #entropy: H(Y) = - sum [ P(Y=i) log2 P(Y=i)]
        # for a binary attribute = -p+log2p+ - p-log2p-

        e = 0.0  #initialize 

        #first need to find the probability
        counter = Counter(Y) 

        for count in counter.values():  #calculate for each unique label
            probability = count / len(Y)
            e -= probability * math.log2(probability)  #subtract the calculated value from e


        #########################################
        return e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # H(Y|X) = sum values(X) [(Sv) /(S)]* Entropy(Sv)
        # values(X): possible values for X
        # Sv: subset of S for which X has value v

        ce = 0.0  #initialize 

        counter = Counter(X)  #get a list of counts of unique values
        total = len(X)

        #work through the different labels
        for x, count in counter.items():
            Sv=[]
            for i, val in enumerate(X):
                if val == x:
                    Sv.append(i) #add the index to Sv

            #get the corresponding Y labels
            S = Y[Sv]

            #use entropy function to calculate the entropy for this subset
            entropy_Sv = Tree.entropy(S)

            #calculate conditional entropy --> add to the running sum ce
            ce += (len(S)/ len(Y)) * entropy_Sv

        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        #information gain = entropy - conditional entropy

        g = 0.0  #initialize

        g= Tree.entropy(Y) - Tree.conditional_entropy(Y,X) #use entropy and conditional entropy to calculate g

 
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        #initialize as low values
        best_g = 0.0
        i = 1

        #loop through each attribute and calculate information gain
        for index in range(X.shape[0]):
            attribute_values = X[index,:]

            #calculate information_gain for the attribute
            info_gain= Tree.information_gain(Y, attribute_values)

            #update best_g
            if info_gain > best_g:
                best_g = info_gain
                i = index

            #break ties
            elif info_gain == best_g and index < i:
                i = index

 
        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        
        C = {} #initialize dictionary

        #get the unique values for the ith attribute
        unique_i_values = np.unique(X[i,:])

        for value in unique_i_values:
            #(1) split the matrix X based upon the values in i-th attribute
            #get the indices where the unique value is
            indices = np.where(X[i,:] == value)

            #create a submatrix with those values
            sub_X = X[:, indices].reshape(X.shape[0], -1) #only want to keep that column

            #(2) split the labels Y based upon the values in i-th attribute
            sub_Y = Y[indices]

            # (4) build the dictionary to combine each  value in the i-th attribute with a child node.
            C[value] = Node(sub_X, sub_Y)
    
        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        unique_labels = np.unique(Y) #get the number of unique labels
        if len(unique_labels) == 1: #check if there's only one unique label
            s = True
        
        else:
            s= False


        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        s = True  #default initalize s to true
        for i in range(1, X.shape[1]):
            if not np.array_equal(X[:, i],X[:, 0]): #check if its equal to the first instance --> if so we can break
                s = False
                break  #break as soon as we know its false
        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        counter = Counter(Y)  #get a list of counts of unique values
        max_count = 0

        for label, count in counter.items(): #find the most common label and update variables
            if count > max_count:
                max_count = count
                y = label

        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        #check stopping conditions first
        if Tree.stop1(t.Y) == True:
            t.isleaf = True
            t.p = Tree.most_common(t.Y)  #set the label to be the most common label (the only label)
            return
        
        if Tree.stop2(t.X) ==True:
            t.isleaf = True
            t.p = Tree.most_common(t.Y) #set the label
            return
        
        if not t.isleaf:  #make sure theres a label
           t.p = Tree.most_common(t.Y)
        
        #if we aren't at a leaf node we need to split and continue making the tree
        #find the best attribute to split on
        t.i = Tree.best_attribute(t.X, t.Y)

        #split on that best attribute--> make the new child node
        t.C = Tree.split(t.X, t.Y, t.i)

        #recursively build tree --> get the the child node and run build_tree on that node
        for child_node in t.C.values():
            Tree.build_tree(child_node)

   
 
        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y) 
        Tree.build_tree(t)  #train by running build_tree

        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        

        #systematically work through the tree until we hit a leaf node
        y = t.p #default set y to be a label

        #first check if t is a leaf node --> if so we know the label
        if t.isleaf == True:
            y= t.p
       

        else:
           #if it isn't a leaf node then get the attribute thats being tested
            x_value = x[t.i]

            #check if the child node exists for that attribute and run inference on that node
            if x_value in t.C:
                child_node = t.C[x_value]
                y= Tree.inference(child_node, x)


        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        #initalize an empty matrix to hold the labels --> make it the shape of X rows
        Y = np.empty(X.shape[1], dtype=object)

        for i in range (X.shape[1]): #for each instance in X make an inference about the label
            Y[i] = Tree.inference(t, X[:,i])

        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        with open(filename, mode='r') as file:
            data_file = csv.reader(file) #use csv from python standard library

            headers = next(data_file)  #skip the headers
            attributes = []
            labels = []

            for col in data_file:
                attributes.append(col[1:])   #add rest of columns to X
                labels.append(col[0])  #add target column to Y

            #turn into arrays
            X = np.array(attributes).T #transpose to get the correct shape
            Y = np.array(labels)

        #########################################
        return X,Y