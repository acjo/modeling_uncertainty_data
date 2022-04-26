#!/usr/bin/env python3
"""
Random Forest Lab

Caelan Osman
Math 403 Sec. 1
Sept. 28, 2021
"""
import graphviz
import time
import os
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        #check to make sure the sample column is greater than the value
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    #create our mask
    mask = np.array([question.match(row) for row in data])

    #get the left and right branches
    matches = data[mask]
    no_match = data[~mask]

    #return nontype if necessary
    if matches.size == 0:
        matches = None
    if no_match.size == 0:
        no_match = None

    return matches, no_match

#Problem 2
def gini(data):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""

    #if data is None return 1
    if data is None:
        return 1
    #get the labels
    labels = data[:, -1]
    #get the size
    N = labels.size
    #find unrepeated_labels
    unrepeated_labels = set(labels)
    #get all the fk values
    f_k = np.array([np.sum([unique_label == label for label in labels ]) / N for unique_label in unrepeated_labels])
    return 1 - np.sum(f_k**2)

def info_gain(left,right,G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""

    #check for nontype and set values
    if left is None:
        D1 = 0
    else:
        D1 = left.shape[0]

    if right is None:
        D2 = 0
    else:
        D2 = right.shape[0]

    #total D is the sum of the two
    D = D1 + D2
    #get gini values for left and right
    G1 = gini(left)
    G2 = gini(right)

    return G - (D1*G1/D + D2*G2/D)

# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""

    #get Gini data
    G = gini(data)
    #find the number of columns
    _, n = data.shape
    n -= 1
    cols = np.arange(n)

    if random_subset:
        new_array = np.random.choice(cols, size=int(np.sqrt(n)), replace=False)
        cols = new_array

    #present max gain and qeustion
    max_gain = -np.infty
    best_question = None
    #iterate through all the columns(but the last)
    for column in cols:
        #iterate through all unique values
        for value in set(data[:, column]):
            #instantiate the current question
            question = Question(column, value, feature_names)
            #get partition
            partitions = partition(data, question)
            #check for nontype
            if partitions[0] is None or partitions[-1] is None:
                continue
            #check for leaf size
            if partitions[0].shape[0] < min_samples_leaf or partitions[-1].shape[0] < min_samples_leaf:
                continue
            #get the temporary info gain
            temp_gain = info_gain(partitions[0], partitions[-1], G)
            #check if temporary info gain is larger than max current
            if temp_gain > max_gain:
                max_gain = temp_gain
                best_question = question

    return max_gain, best_question

# Problem 4
class Leaf:
    """Tree leaf node
    Attribute
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        #get unique labels
        labels = set(data[:, -1])
        #map label to number of labels
        self.prediction = {label: sum(data[:, -1] == label) for label in labels}

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, right_branch, left_branch):
        #set attributes
        self.question = question
        self.right = right_branch
        self.left = left_branch

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""


    #check current depth
    if current_depth >= max_depth:
        return Leaf(data)
    #check leaf size
    elif data.shape[0] < min_samples_leaf:
        return Leaf(data)

    #get info gain and best question
    infGain, best_question = find_best_split(data, 
    feature_names, 
    min_samples_leaf = min_samples_leaf, 
    random_subset=random_subset)
    #if the info gain is zero return a leaf
    if infGain == 0:
        return Leaf(data)

    #if the best question is non return a leaf
    if best_question is None:
        return Leaf(data)

    #get left and right partition
    left, right = partition(data, best_question)
    #recursively call left and right node
    left_node = build_tree(left, 
    feature_names, 
    min_samples_leaf = min_samples_leaf, 
    max_depth=max_depth, 
    current_depth =current_depth + 1, 
    random_subset=random_subset)

    right_node = build_tree(right,
    feature_names,
    min_samples_leaf = min_samples_leaf,
    max_depth=max_depth, 
    current_depth =current_depth + 1, 
    random_subset=random_subset)
    #return the decision Node
    return Decision_Node(best_question, right_node, left_node)

# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    #if the type is a Leaf return the prediction
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)

    #if the type is a decision node
    if isinstance(my_tree, Decision_Node):
        #does the sample match?
        match = my_tree.question.match(sample)
        #if so recursively call predict tree with the left side
        if match:
            return predict_tree(sample, my_tree.left)
        #if not recursively call predict tree with the right side
        else:
            return predict_tree(sample, my_tree.right)


def analyze_tree(dataset, my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    #get number of rows of dataset
    m, _ = dataset.shape
    #make predictions for each row
    predictions = np.array([predict_tree(dataset[i], my_tree) for i in range(m)])
    #get the number of correct
    correct = sum(predictions == dataset[:, -1])
    #return the ratio
    return correct/m

# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""

    #get the labels
    predicted_labels = [predict_tree(sample, tree) for tree in forest]
    #return the most occuring predicted label
    return max(set(predicted_labels), key=predicted_labels.count)

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""

    #number of samples
    m, _ = dataset.shape
    #get predictions
    predictions = np.array([predict_forest(dataset[i], forest) for i in range(m)])
    #get the number that are correct
    correct = sum(predictions == dataset[:, -1])
    #return ratio
    return correct/m


# Problem 8
def prob8():
    """Use the file parkinsons.csv to analyze a 5 tree forest.

    Create a forest with 5 trees and train on 100 random samples from the dataset.
    Use 30 random samples to test using analyze_forest() and SkLearn's
    RandomForestClassifier.

    Create a 5 tree forest using 80% of the dataset and analzye using
    RandomForestClassifier.

    Return three tuples, one for each test.

    Each tuple should include the accuracy and time to run: (accuracy, running time)
    """

    #load in the parkinsons data
    parkinsons = np.loadtxt('parkinsons.csv', delimiter=',')[:, 1:]
    #load in the features
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str)


    #get sample sizes
    random_sample_size = 130
    random_training_size = 100
    #shuffle the data and define the sample and fit to use for the forest
    np.random.shuffle(parkinsons)
    random_samples = parkinsons[:random_sample_size]
    random_training = random_samples[:random_training_size]
    random_fit = random_samples[random_training_size:]

    #build the forrest and time it and test accuracy
    start = time.time()
    my_forest = [build_tree(random_training, features, min_samples_leaf=15, random_subset=True) for _ in range(5)]
    end = time.time()
    my_time_to_build = np.abs(end-start)
    my_accuracy = analyze_forest(random_fit, my_forest)


    #build the same forest using scikitlearn.
    #Time it and get accuracy
    start = time.time()
    their_forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    their_forest.fit(random_training[:, :-1], random_training[:, -1])
    end = time.time()
    their_time_to_build = np.abs(end-start)
    their_accuracy = their_forest.score(random_fit[:, :-1], random_fit[:, -1])


    #now build the entire forest
    m, _ =  parkinsons.shape
    training_amount = int(0.8*m)
    start = time.time()
    entire_forest = RandomForestClassifier()
    entire_forest.fit(parkinsons[training_amount:, :-1], parkinsons[training_amount:, -1])
    end = time.time()
    entire_time_to_build = np.abs(end-start)
    entire_accuracy = entire_forest.score(parkinsons[:training_amount, :-1], parkinsons[:training_amount, -1])

    return (my_accuracy, my_time_to_build), (their_accuracy, their_time_to_build), (entire_accuracy, entire_time_to_build)


if __name__ == "__main__":
    #print(prob8())
    #pass
    features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
    animals = np.loadtxt('animals.csv', delimiter=',' )


    tree = build_tree(animals, features, random_subset=False)
    draw_tree(tree)
    '''

    #tree = build_tree(animals[:80], features)

    #print(analyze_tree(animals[80:], tree))

    num = 1000
    val = 0
    for _ in range(num):
        np.random.shuffle(animals)
        forrest = [build_tree(animals[:80], features, random_subset=True) for _ in range(4)]
        val += analyze_forest(animals[80:], forrest)
    print(val/num)

    names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)
    print(gini(animals))
    print(info_gain(animals[:50], animals[50:], gini(animals)))

    print(find_best_split(animals, features))
    '''
