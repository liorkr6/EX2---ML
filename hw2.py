import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """

    ###########################################################################
    class_column = data[:, -1]
    total_entries = len(class_column)
    classes, counts = np.unique(class_column, return_counts=True)
    gini = 1
    for c in counts:
        prob = c / total_entries
        gini -= prob ** 2
    return gini

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    class_column = data[:, -1]
    total_entries = len(class_column)
    classes, counts = np.unique(class_column, return_counts=True)
    for c in counts:
        prob = c / total_entries
        entropy -= prob * np.log2(prob)
    return entropy

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    split = 1
    if gain_ratio:
        impurity_func = calc_entropy
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    feature_col = data[:, feature]
    total_entries_feature = len(feature_col)
    father_val = impurity_func(data)
    children_val = 0

    options_of_feature, values = np.unique(feature_col, return_counts=True)
    feature_map = dict(zip(options_of_feature, values))

    for feature_name, value in feature_map.items():
        relevant_data = feature_col == feature_name
        groups[feature_name] = data[relevant_data]
        value_prob = value / total_entries_feature
        children_val += value_prob * impurity_func(groups[feature_name])
        if gain_ratio:
            split -= value_prob * np.log2(value_prob)

    goodness = (father_val - children_val) / split

    return goodness, groups

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """

        pred_options, counts = np.unique(self.data[:, -1], return_counts=True)
        pred = pred_options[np.argmax(counts)]
        return pred

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """

        if impurity_func(self.data) == 0 or self.depth >= self.max_depth:
            self.terminal = True
            return

        best_feature_index = None
        best_feature_val = 0
        best_feature_data = None
        features_num = np.shape(self.data)[1] - 1

        for feature_index in range(features_num):
            potential_val, potential_data = goodness_of_split(self.data, feature_index, impurity_func, self.gain_ratio)
            if potential_val > best_feature_val:
                best_feature_index = feature_index
                best_feature_val = potential_val
                best_feature_data = potential_data

        if best_feature_val == 0:
            self.terminal = True
            return

        self.feature = best_feature_index

        for item_data in best_feature_data:
            child_node = DecisionNode(data=best_feature_data[item_data], depth=self.depth + 1, max_depth=self.max_depth,
                                      gain_ratio=self.gain_ratio, chi=self.chi)
            self.add_child(child_node, item_data)
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """

    def split_recursive(node, impurity):
        if node.terminal:
            return

        node.split(impurity)
        for child in node.children:
            split_recursive(child, impurity)

    root = DecisionNode(data=data, max_depth=max_depth, gain_ratio=gain_ratio, chi=chi)
    split_recursive(root, impurity)
    return root
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """

    if not root.terminal:
        for child in root.children_values:
            if child == instance[root.feature]:
                feature = root.children_values.index(child)
                return predict(root.children[feature], instance)
    return root.pred

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    failures = 0
    for row in dataset:
        prediction = predict(node, row)
        label = row[-1]
        if prediction != label:
            failures += 1
    failures_percentage = failures / len(dataset)
    accuracy = (1 - failures_percentage) * 100
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
