from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

iris.keys()

data = iris.data

target = iris.target

x_label = 1

y_label = 2

c_dict = {0:'red', 1:'blue', 2:'green'}

for type in range(len(iris.target_names)):
    plt.scatter(data[type==target, x_label],
               data[type==target,y_label],
                c = c_dict[type],
               label=iris.target_names[type])

plt.x_label=iris.feature_names[x_label]
plt.y_label=iris.feature_names[y_label]
plt.legend(loc='upper left')
plt.show()

df = pd.DataFrame(data,columns=iris.feature_names)

pd.scatter_matrix(df, alpha=0.5, figsize=(8,8));

x, y = iris.data, iris.target
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=.5, test_size=.5, random_state=123, stratify=y)

print(np.bincount(y)/float(len(y)))
print(np.bincount(train_y)/float(len(train_y)))
print(np.bincount(test_y)/float(len(test_y)))

model = KNeighborsClassifier().fit(train_x, train_y)

pred = model.predict(test_x)

print(np.sum(pred==test_y)/float(len(test_y)))

x=3
y=2
c_text = {0:'red', 1:'blue', 2:'green', 3:'yellow'}
for i in np.unique(test_y):
    mp.scatter(test_x[test_y==i,x], test_x[test_y==i,y], c=c_text[i])
mp.scatter(test_x[test_y!=pred,x], test_x[test_y!=pred,y], c=c_text[3])
mp.show()

x = np.where(pred!=test_y)
x
test_x[x][:,[2,3]]
