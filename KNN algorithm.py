import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    '''
    Predicts the label/diagnosis of a set of features

    INPUT
        data: features of the training dataset
        predict: features to be diagnosed

    OUTPUT
        return a predicted label
    '''

    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting group!')
    distances = []
    for group in data:
        for feature in data[group]:
            # calculating distance between training data and testing data
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])

    # collecting the classes of the top k features with the shortest distance
    votes = [i[1] for i in sorted(distances)[:k]]
    # taking the class with the highest vote
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


df = pd.read_csv('breast-cancer-dataset.txt')
df.replace("?", -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()  # converting non numeric data
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        votes = k_nearest_neighbors(train_set, data, k=5)
        if group == votes:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy}")
