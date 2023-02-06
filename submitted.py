'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    distance = [np.linalg.norm(image - train_image) for train_image in train_images]
    nearest_indices = np.argsort(distance)[:k]
    neighbors = [train_images[i] for i in nearest_indices]
    labels = [train_labels[i] for i in nearest_indices]
    return np.array(neighbors), np.array(labels)

    #raise RuntimeError('You need to write this part!')


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    
    distances = np.sqrt(np.sum((dev_images[:, np.newaxis, :] - train_images) ** 2, axis=-1))
    indices = np.argpartition(distances, kth=k, axis=-1)[:, :k]
    knn_labels = np.take(train_labels, indices)
    majority_votes = np.argmax(np.apply_along_axis(np.bincount, axis=1, arr=knn_labels), axis=-1)
    hypotheses = majority_votes.tolist()
    scores = np.sum(knn_labels == majority_votes[:, np.newaxis], axis=-1).tolist()
        
    return hypotheses, scores
    
    #raise RuntimeError('You need to write this part!')


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    # Find the unique labels
    labels = list(set(references))
    
    # Create the empty confusion matrix with shape (num_labels, num_labels)
    confusions = np.zeros((len(labels), len(labels)), dtype=int)
    
    # Loop through the hypotheses and references and update the confusion matrix
    for i in range(len(hypotheses)):
        h_index = labels.index(hypotheses[i])
        r_index = labels.index(references[i])
        confusions[r_index][h_index] += 1

    # Calculate the accuracy
    accuracy = np.sum(np.diagonal(confusions)) / len(hypotheses)
    
    # Calculate the f1 score
    precision = np.zeros(len(labels))
    recall = np.zeros(len(labels))
    TP = confusions[1][1]
    FP = confusions[0][1]
    FN = confusions [1][0]
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * np.mean(precision * recall / (precision + recall))

    return confusions, accuracy, f1

    #raise RuntimeError('You need to write this part!')
