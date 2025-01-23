import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

print("Loading dataset")
train_data = np.load("Data/fashion_train.npy")
test_data = np.load("Data/fashion_test.npy")
print("Dataset loaded")

X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

X_train = X_train / 255.0
X_test = X_test / 255.0

clf = SklearnDecisionTree(max_depth=9)  
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=[f'Pixel {i}' for i in range(X_train.shape[1])], class_names=[str(i) for i in np.unique(y_train)], rounded=True, precision=3)
plt.title("Decision Tree Visualization")
plt.show()
