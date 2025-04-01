import numpy as np
import matplotlib.pyplot as plt

def path_length(tree, point, current_depth):
    if 'value' in tree:
        return current_depth
    feature, threshold = tree['feature'], tree['threshold']
    if point[feature] <= threshold:
        return path_length(tree['left'], point, current_depth + 1)
    else:
        return path_length(tree['right'], point, current_depth + 1)

def build_tree(data, max_depth):
    if len(data) <= 1 or max_depth <= 0:
        return {'value': data}
    feature = np.random.randint(data.shape[1])
    min_val, max_val = data[:, feature].min(), data[:, feature].max()
    if min_val == max_val:
        return {'value': data}
    threshold = np.random.uniform(min_val, max_val)
    left = data[data[:, feature] <= threshold]
    right = data[data[:, feature] > threshold]
    return {
        'feature': feature,
        'threshold': threshold,
        'left': build_tree(left, max_depth - 1),
        'right': build_tree(right, max_depth - 1)
    }

def isolation_forest(data, n_trees=100, max_samples=256):
    forest = []
    max_depth = np.ceil(np.log2(min(max_samples, len(data))))
    for _ in range(n_trees):
        subsample = data[np.random.choice(len(data), min(max_samples, len(data)), replace=False)]
        tree = build_tree(subsample, max_depth)
        forest.append(tree)
    return forest

def anomaly_score(avg_path_length, n):
    c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 1 else 1
    return 2 ** (-avg_path_length / c)

def compute_scores(forest, data, n):
    scores = []
    for point in data:
        path_lengths = [path_length(tree, point, 0) for tree in forest]
        avg_path_length = np.mean(path_lengths)
        score = anomaly_score(avg_path_length, n)
        scores.append(score)
    return np.array(scores)


