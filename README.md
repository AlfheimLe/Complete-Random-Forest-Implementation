
# 1. Implementing a Decision Tree


```python
import numpy as np
import scipy.io
from scipy import stats
import random
import matplotlib.pyplot as plt
```


```python
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.label = None
        self.split_rule = None
        self.height = 0
```


```python
class DecisionTree:

    def __init__(self, MAX_HEIGHT, NODE_PURITY, INFORMATION_GAIN, features_random = 32): #X.shape[1] = 32
        """
        initialization of a decision tree
        """
        self.Node = Node()
        self.MAX_HEIGHT = MAX_HEIGHT
        self.NODE_PURITY = NODE_PURITY
        self.INFORMATION_GAIN = INFORMATION_GAIN
        self.features_random = features_random
        
    @staticmethod
    def entropy(y):
        """
        entropy given all the labels
        """
        if len(y)==0:
            return 0
        p_1 = sum(y)/len(y)
        p_0 = 1 - p_1
        
        if p_1 == 0:
            A = 0
        else:
            A = -p_1*np.log10(p_1)
        if p_0 == 0:
            B = 0
        else:
            B = - p_0*np.log10(p_0)
        H = A+B
        return H

    @staticmethod
    def information_gain(X, y, thresh):
        """
        information gain given a vector of features
        and a split threshold
        """
        T = np.array([(x<thresh) for x in X]).T
        F = T.copy()
        T = True^T
        y_T = np.ma.masked_array(y, T).compressed()
        y_F = np.ma.masked_array(y, F).compressed()
        T = 1 - sum(T)/len(T)
        F = 1 - T
        H = T*DecisionTree.entropy(y_T)+F*DecisionTree.entropy(y_F)
        IG = DecisionTree.entropy(y) - H
        return IG

    @staticmethod
    def gini_impurity(y):
        """
        gini impurity given all the labels
        """
        if len(y)==0:
            return 0
        p_1 = sum(y)/len(y)
        p_0 = 1 - p_1
        return 1 - p_1**2 - p_0**2

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        T = np.array([(x<thresh) for x in X]).T
        F = T.copy()
        T = True^T
        y_T = np.ma.masked_array(y, T).compressed()
        y_F = np.ma.masked_array(y, F).compressed()
        T = 1 - sum(T)/len(T)
        F = 1 - T
        G = T*DecisionTree.gini_impurity(y_T)+F*DecisionTree.gini_impurity(y_F)
        return DecisionTree.gini_impurity(y) - G

    def split(self, X, y, idx, thresh):
        """
        return a split of the dataset given an index of the feature and
        a threshold for it
        """
        T = np.array([(x<thresh) for x in X[:,idx]])
        T = np.array([T for _ in range(X.shape[1])]).T
        F = T.copy()
        T = True^T

        X_A = np.ma.compress_rows(np.ma.masked_array(X, T))
        X_B = np.ma.compress_rows(np.ma.masked_array(X, F))
        y_A = np.ma.masked_array(y, T[:,0]).compressed()
        y_B = np.ma.masked_array(y, F[:,0]).compressed()
        return [[X_A, y_A],[X_B, y_B]]
    
    def segmenter(self, X, y):
        """
        compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        features = np.random.choice(X.shape[1], self.features_random, replace = False)
            
        gain = np.zeros((X.shape))
        for j in features:
            unique = np.unique(X[:,j])
            gain_unique = dict()
            for i in range(len(unique)):
                thresh = unique[i]
                gain_unique[unique[i]] = DecisionTree.information_gain(X[:,j],y,thresh)
            for i in range(1,X.shape[0]):
                gain[i,j] = gain_unique[X[i,j]]
        idx = np.unravel_index(np.argmax(gain, axis=None), gain.shape)
        return [idx[1], X[idx[0], idx[1]]] 
    
    
    def train(self, X, y, node=None):
        """
        fit the model to a training set.
        """
        if node == None:
            self.Node = Node()
            node = self.Node
        if node.height > self.MAX_HEIGHT or np.sum(y)/len(y) > self.NODE_PURITY or np.sum(y)/len(y) < 1-self.NODE_PURITY or len(X)<=1:
            node.label = int(np.sum(y)/len(y) >= 0.5)
        else:
            idx, thresh = self.segmenter(X,y)
            A, B = self.split(X,y,idx,thresh)
            X_A, y_A = A
            X_B, y_B = B
            IG = DecisionTree.information_gain(X[:,idx], y, thresh)
            if IG >= self.INFORMATION_GAIN and len(y_A)>0 and len(y_B)>0:
                node.split_rule = [idx, thresh]
                
                
                node.left = Node()
                node.left.height = node.height + 1
                self.train(X_A, y_A, node.left)
                node.right = Node()
                node.right.height = node.height + 1
                self.train(X_B, y_B, node.right)
            else:
                node.label = int(np.sum(y)/len(y) >= 0.5)   

    def predict_single(self, X, node=None):
        """
        predict the labels for input data 
        """
        if node == None:
            node = self.Node
        if node.label != None:
            #print('Therefore, this email is: ', class_names[node.label]+'\n')
            return node.label
        else:
            idx, thresh = node.split_rule
            if X[idx] < thresh and node.left != None:
                #print(features[idx] + ' < ' + str(thresh)+'\n')
                return self.predict_single(X, node.left)
            elif X[idx] >= thresh and node.right != None:
                #print(features[idx] + ' >= ' + str(thresh)+'\n')
                return self.predict_single(X, node.right)
                
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return predictions

    def __repr__(self, node=None, indent=0):
        """
        String representation of the tree
        """
        if node == None:
            node = self.Node
        if node.label == None:
            idx, thresh = node.split_rule
            print('  '*indent + str(features[idx])+': '+'x < '+str(thresh))
            if node.left!=None:
                self.__repr__(node.left, indent + 1)
            if node.right!=None:
                self.__repr__(node.right, indent + 1)
        else:
            print('  '*indent + 'label: ' + str(class_names[node.label]))
            
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = [y_test[i] == predictions[i] for i in range(len(y_test))]
        if len(accuracy)==0:
            return 0
        return sum(accuracy)/len(accuracy)
```

# 2. Using the Decision Tree class to implement a Random Forest


```python
class RandomForest():
    
    def __init__(self, MAX_HEIGHT, NODE_PURITY, INFORMATION_GAIN, features_randomization, number_of_trees, subsample):
        """
        TODO: initialization of a random forest
        """
        self.subsample = subsample
        self.forest = [DecisionTree(MAX_HEIGHT, NODE_PURITY, INFORMATION_GAIN, features_randomization) for _ in range(number_of_trees)]

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        for tree in self.forest:
            subsample = np.random.choice(len(X), self.subsample)
            X_train = np.array([X[i] for i in subsample])
            y_train = np.array([y[i] for i in subsample])
            tree.train(X_train, y_train)
        
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions).T
        return np.array([(sum(pred)/len(pred) >= 0.5) for pred in predictions])
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = [y_test[i] == predictions[i] for i in range(len(y_test))]
        if len(accuracy)==0:
            return 0
        return sum(accuracy)/len(accuracy)
```

# 3. Implementing k-fold Cross Validation


```python
def k_fold_cv(model, X, y, k=10):
    training_acc = 0
    testing_acc = 0
    cpt = 0
    step = len(X)//k
    for i in range(0,k):
        if i==k-1:
            X_test = X[i*step::]
            y_test = y[i*step::]
            X_train = X[0:i*step]
            y_train = y[0:i*step]
        else:
            X_test = X[i*step:(i+1)*step]
            y_test = y[i*step:(i+1)*step]
            X_train = np.concatenate((X[0:i*step],X[(i+1)*step::]))
            y_train = np.concatenate((y[0:i*step],y[(i+1)*step::]))
        
        model.train(X_train, y_train)
        model_train_acc = model.evaluate(X_train, y_train)
        training_acc += model_train_acc
        model_test_acc = model.evaluate(X_test, y_test)
        testing_acc += model_test_acc
        print('Training {} of {}: - Training accuracy: {} - Testing accuracy: {}'.format(i+1, k, model_train_acc, model_test_acc))
    print('\nAverage Training Accuracy: {} / Average Testing Accuracy: {}'.format(training_acc/k, testing_acc/k))
    return training_acc/k, testing_acc/k
```

## Given a dataset, one can plot the influence of the depth of a decision tree


```python
    for i in range(1,40):
        depth.append(i)
        classifier = DecisionTree(i, 1, 0.0)
        classifier.train(X_train, y_train)
        train = classifier.evaluate(X_train, y_train)
        test = classifier.evaluate(X_test, y_test)
        training_acc.append(train)
        testing_acc.append(test)
    plt.plot(depth, testing_acc, label='Validation Accuracy')
    plt.plot(depth, training_acc, label='Training Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a Function of the Depth')
    plt.legend()
    plt.show();
```

# 3. Training and Visualizing a Classifier


```python
classifier = DecisionTree(5, 1, 0.0)
training_accuracy, testing_accuracy = k_fold_cv(classifier, X, y, 5) #to get an idea of the classifier's accuracy

classifier.train(X,y)
classifier.__repr__()
```
