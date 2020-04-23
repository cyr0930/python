from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from machine_learning import utils, data

svm = SVC(kernel='linear', C=1.0, random_state=1)
# svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=1)   # Kernel trick
# svm = SGDClassifier(loss='hinge')   # for large dataset
svm.fit(data.X_train_std, data.y_train)
utils.plot_decision_regions(data.X_combined_std, data.y_combined, classifier=svm, test_idx=range(105, 150))
