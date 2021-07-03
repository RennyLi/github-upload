class Node:
    def __init__(self,value=None,trueBranch=None,falseBranch=None,results=None,col=-1,data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.data = data

class Tree:
    def __init__(self):
        self.head = None
        self.leaf = None

    def add_head(self,col,value,trueBranch,falseBranch):
        self.head = Node(col=col, value=value,
                         trueBranch=trueBranch, falseBranch=falseBranch)
        return self.head

    def add_leaf(self,results,data):
        self.leaf = Node(results=results, data=data)
        return self.leaf

    def find_head(self):
        return self.head


def readtrain():
    """  Read the train dataset.  """
    f = open('d://xxxx//Python//CSC1001//project//train.csv', 'r')
    lines = []
    for line in f:
        tmpt = line.strip().split(", ")
        lines.append(tmpt)
    dataset = []
    for line in lines[1:]:
        new_line = []
        for i in line[:-1]:
            new_line.append(float(i))
        if float(line[-1]) > 6:
            new_line.append(True)
        else:
            new_line.append(False)
        dataset.append(new_line)
    f.close()
    return dataset

def readtest():
    """  Read the test dataset.  """
    f = open('d://xxxx//Python//CSC1001//project//test.csv', 'r')
    lines= []
    for line in f:
        tmpt = line.strip().split(", ")
        lines.append(tmpt)
    test_data = []
    for line in lines[1:]:
        new_set = []
        for i in line:
            new_set.append(float(i))
        test_data.append(new_set)
    return test_data


def calculateDiffCount(dataset):
    """  Classify the elements in the dataset.  """
    # return {type1: count1,type2: count2 ,..., typeN: countN}
    results = {}
    for data in dataset:
        if data[-1] not in results:
            results[data[-1]] = 1
        else:
            results[data[-1]] += 1
    return results

def gini(rows):
    """  Calculate gini.  """
    length = len(rows)
    results = calculateDiffCount(rows)
    impurity = 0.0
    for i in results.keys():
        impurity += (results[i] / length)**2
    return 1-impurity

def splitDataset(rows,value,column):
    """  Spilt the dataset according to the value and column.  """
    trueList = [] # Contain all the rows that satisfy the characteristic condition
    falseList = [] # Contain all the rows that do not satisfy the characteristic condition
    for row in rows:
        if row[column] >= value:
            trueList.append(row)
        else:
            falseList.append(row)
    return trueList, falseList

def buildDecisionTree(rows):
    """  Build the decision tree.  """
    currentGain = gini(rows)
    columnNumber = len(rows[0])
    rowNumber = len(rows)

    bestGain = 0.0
    bestValue = None
    bestSet = None

    # The best gain is to minimize the gini
    for col in range(columnNumber - 1):  # Traverse the different eigenvalues in the dataset
        col_value_set = set([x[col] for x in rows])  # Clear repeated eigenvalues
        for value in col_value_set:
            trueList, falseList = splitDataset(rows, value, col)
            p = len(trueList) / rowNumber
            gain = currentGain - p * gini(trueList) - (1-p) * gini(falseList) # Use Gini Gain to find the best eigenvalue
            if gain > bestGain:
                bestGain = gain
                bestValue = (col,value) # Col represents the column in which the eigenvalues are located
                bestSet = (trueList,falseList) 

    # Use recursion to build the tree.
    if bestGain > 0:
        trueBranch = buildDecisionTree(bestSet[0])
        falseBranch = buildDecisionTree(bestSet[1])
        return decisionTree.add_head(bestValue[0], bestValue[1], 
                    trueBranch, falseBranch)
    else:  # bestGain = 0 means that the dataset can no longer be syncopated, so the recursion will stop.
        return decisionTree.add_leaf(calculateDiffCount(rows), rows) 
                # The last node of Tree, recording the ddistribution of wine quality after the test data

def prune(head, miniGain):
    """  If gain < mini gain, combine trueBranch with falseBranch.  """
    if head.trueBranch.results == None:
        prune(head.trueBranch, miniGain)
    
    if head.falseBranch.results == None:
        prune(head.falseBranch, miniGain)
     
    if head.trueBranch.results != None and head.falseBranch.results != None:
        len1 = len(head.trueBranch.data)
        len2 = len(head.falseBranch.data)

        p = float(len1) / (len1 + len2)

        gain = gini(head.trueBranch.data + head.falseBranch.data) \
                - p * gini(head.trueBranch.data) \
                - (1-p) * gini(head.falseBranch.data)

        if gain < miniGain:
            head.data = head.trueBranch.data + head.falseBranch.data # Combine the data
            head.results = calculateDiffCount(head.data)
            head.trueBranch = None
            head.falseBranch = None

def classify(data, head):
    """  Classify the testing data.  """
    # Get the classification
    if head.results != None:
        return head.results # Return the reslut after index (True of False)
    # Find the classification
    else:
        branch = None
        v = data[head.col] # head.col means the position of the optimal index
        if v >= head.value: 
            branch = head.trueBranch # Meet the characteristic conditions
        else:
            branch = head.falseBranch # Do not meet the characteristic conditions
        return classify(data,branch) # Continually recursive until get the result node of the Tree

if __name__ == '__main__':
    # Get the train dataset.
    dataset = readtrain()
    # Build the decision tree.
    decisionTree = Tree()
    buildDecisionTree(dataset)
    head = decisionTree.find_head()
    prune(head, 0.4)
    # Get the test dataset.
    test_data = readtest()
    # Classify the test data
    total = 0
    count = 0
    for row in test_data:
        r = classify(row, head)
        for key,value in r.items():
            if value == max(r.values()): # After classfication, r may contain two keys: True and False.
                                         # We choose the one which has large amount to be the result of prediction.
                if key == (float(test_data[total][-1])>6):
                    print(key, end=' ')  # True means the quality score is above 6. False means the contrary.
                    print("Correct")
                    count += 1
                    break
                else:
                    print(key, end=' ')
                    print("Incorrect")
                    break
        total += 1
    print()
    print(count)
    print(total)
    print("The accuracy is:" + str(count/total))