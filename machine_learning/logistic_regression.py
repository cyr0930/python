from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from machine_learning import utils, data


lr = LogisticRegression(solver='liblinear', C=100.0, random_state=1)    # One-versus-rest
# lr = SGDClassifier(loss='log')  # for large dataset
lr.fit(data.X_train_std, data.y_train)

y_pred = lr.predict(data.X_test_std)
print('\nLabel\tProbability')
for i, p in enumerate(lr.predict_proba(data.X_test_std[:3, :])):
    print(f'{y_pred[i]}\t\t{p}')
print('\nMisclassified:', (data.y_test != y_pred).sum())
print('Accuracy:', accuracy_score(data.y_test, y_pred))

utils.plot_decision_regions(X=data.X_combined_std, y=data.y_combined, classifier=lr, test_idx=range(105, 150))
