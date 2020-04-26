import numpy as np
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from machine_learning import utils, data

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(data.X_train_std, data.y_train)
utils.plot_decision_regions(data.X_combined_std, data.y_combined, classifier=tree, test_idx=range(105, 150))

class_names = ['Setosa', 'Versicolor', 'Virginica']
feature_names = ['petal length', 'petal width']
dot_data = export_graphviz(tree, filled=True, rounded=True, out_file=None,
                           class_names=class_names, feature_names=feature_names)
graph = graph_from_dot_data(dot_data)
graph.write_png('tmp/tree.png')

forest = RandomForestClassifier(criterion='entropy', n_estimators=16, n_jobs=2, random_state=1)
forest.fit(data.X_train_std, data.y_train)
utils.plot_decision_regions(data.X_combined_std, data.y_combined, classifier=forest, test_idx=range(105, 150))

# Feature selection
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(data.X_train_std)
print()
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 20, feature_names[indices[f]], importances[indices[f]]))
