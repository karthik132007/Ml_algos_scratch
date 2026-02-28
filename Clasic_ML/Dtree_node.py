import numpy as np


"""
    Ear Shape (1 if pointy, 0 otherwise)
    Face Shape (1 if round, 0 otherwise)
    Whiskers (1 if present, 0 otherwise)
"""
X_train = np.array([[1, 1, 1],
[0, 0, 1],      # floppy ears, not round face, whiskers present
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0]) 

class Node:
    def __init__(self,feature=None,left=None,right=None,value=None):
        self.feature = feature
        self.left = left
        self.right= right
        self.value = value

def entropy(y):
    entropy =0
    if len(y)!=0:
        p1= len(y[y==1])/len(y)
        if p1!=1 and p1!=0:
            entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
        else:
            entropy=0
    return entropy


def split_data(X,node_indices,feature):

    left,right =[],[]
    for i in node_indices:
        if X[i][feature]==1:
            left.append(i)
        else:
            right.append(i)
    return left,right

def information_gain(x,y,node_indices,feature):
    left_indices,right_indices = split_data(x,node_indices,feature)

    if len(left_indices) == 0 or len(right_indices)==0:
        return 0
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    node_entropy = entropy(y_node)
    left_entropy= entropy(y_left)
    right_entropy = entropy(y_right)

    w_left=  len(left_indices)/len(node_indices)
    w_right=  len(right_indices)/len(node_indices)

    information_gain = node_entropy - ((w_left*left_entropy) + (w_right*right_entropy))

    return information_gain

def get_best_split(X,Y,node_indices,used_features):
    best_feature =-1
    max_info_gain =0
    num_features = X.shape[1]

    for feature in range(num_features):
        if feature in used_features:
            continue
        info_gain = information_gain(X,Y,node_indices,feature)
        if max_info_gain<info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature


def majority_class(y):
    if np.sum(y==1) >= np.sum(y==0):
        return 1
    else:
        return 0



def build_tree(X,Y,node_indices,used_features):
    y_node= Y[node_indices]

    if len(set(y_node)) ==1:
        return Node(value=y_node[0])   #checking pure node or not
    
    if len(used_features) == X.shape[1]:
        return Node(value=majority_class(y_node))
    
    best_feature = get_best_split(X,Y,node_indices,used_features)

    if(best_feature==-1):
        return Node(value=majority_class(y_node))
    
    left_indices, right_indices = split_data(X,node_indices,best_feature)

    new_used_features = used_features + [best_feature]

    left_child = build_tree(X,Y,left_indices,new_used_features)
    right_child = build_tree(X,Y,right_indices,new_used_features)

    return Node(feature=best_feature,left=left_child,right=right_child)

def predict_one(x,node):
    
    if node.value is not None:
        return node.value
    
    if x[node.feature] ==1:
        return predict_one(x,node.left)
    else:
        return predict_one(x,node.right)
    


def predict(X,node):
    ans = np.array([predict_one(x,node) for x in X])
    return ans


test = [0,1,0]
root_indices=  list(range(len(y_train)))

root = build_tree(X_train, y_train, root_indices,used_features=[])
    
res = predict(X_train,root)

print(res==y_train)