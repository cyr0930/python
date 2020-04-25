from sklearn.neighbors import KNeighborsClassifier
from machine_learning import utils, data

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')  # euclidean distance
knn.fit(data.X_train_std, data.y_train)
utils.plot_decision_regions(data.X_combined_std, data.y_combined, classifier=knn, test_idx=range(105, 150))
