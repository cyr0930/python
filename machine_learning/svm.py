from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from machine_learning import common

svm = SVC(kernel='linear', C=1.0, random_state=1)
# svm = SGDClassifier(loss='hinge')   # for large dataset
svm.fit(common.X_train_std, common.y_train)
common.plot_decision_regions(common.X_combined_std, common.y_combined, classifier=svm, test_idx=range(105, 150))
