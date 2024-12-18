import numpy as np
import csv
from sklearn import neighbors
from sklearn.model_selection import train_test_split

f = open('D:\\python\\lab\\project\\results.csv', 'r')
result_data = csv.reader(f)
result_data_str = np.array(list(result_data))
result_data_str = np.delete(result_data_str, 0, 0) 
result_data_str = np.delete(result_data_str, 0, 1) 

result_data_ = result_data_str.astype(int)



X_data = np.delete(result_data_,[7,8], 1)
y_data = np.delete(result_data_,[0,1,2,3,4,5,6,7], 1).flatten()
# y_data = np.array(y_data_tem, ndmin=1)

# print(X_data)
# print(y_data_tem)
# print(result_data_)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.5, random_state=42)
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'uniform')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score
print ("Accuracy of KNN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
