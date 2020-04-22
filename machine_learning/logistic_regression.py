from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from machine_learning import common


lr = LogisticRegression(solver='liblinear', C=100.0, random_state=1)    # One-versus-rest
# lr = SGDClassifier(loss='log')  # for large dataset
lr.fit(common.X_train_std, common.y_train)

y_pred = lr.predict(common.X_test_std)
print('\nLabel\tProbability')
for i, p in enumerate(lr.predict_proba(common.X_test_std[:3, :])):
    print(f'{y_pred[i]}\t\t{p}')
print('\nMisclassified:', (common.y_test != y_pred).sum())
print('Accuracy:', accuracy_score(common.y_test, y_pred))

common.plot_decision_regions(X=common.X_combined_std, y=common.y_combined, classifier=lr, test_idx=range(105, 150))
