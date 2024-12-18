import numpy as np
from collections import Counter
import csv
from sklearn.model_selection import train_test_split
from typing import List
class KNN:
    def __init__(self, k: int, weight: str):
        """
        Initialize the KNN object.

        Args:
            k(int) : the number of neighbor (nearest dot)
            weight (str): use to chose 2 options (uniforms and ) for weight calculation
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.weight = weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Training data

            Args:
                X(np.ndarray): Output Data
                y(np.ndarray): 
        """
        self.X_train = X
        self.y_train = y

#duyet x trong xtest truyen vao one
    def predict(self, X_test: np.ndarray) -> List[np.ndarray]:
        """
            Predict labels for training set

            Args:
                X_test: features data 

            Return:
                preds:list(int): list of predicted label
            
        
        """
        preds = []
        for i, x in enumerate(X_test):
            # print(f"x_test_{i}: {x}")
            preds.append(self._predict_one(x))
        return preds
        
        
    def _euclidean_distance(self, p, q):
        """

        
        """
        return np.sqrt(np.sum((p - q) ** 2))

    def _manhattan_distance(self, p, q):
        return np.sum(np.abs(p - q))
    
   


#tinh kc cua x voi cac phantu x train
    def _predict_one(self, x):
        _distances = [self._euclidean_distance(x, x_i) for x_i in self.X_train]

        k_index = np.argsort(_distances)[:self.k]
        k_nearest_labels = []
        k_nearest_distances = []
        for i in k_index:
            k_nearest_labels.append(self.y_train[i])
            k_nearest_distances.append(_distances[i])   
        _distances = [self._euclidean_distance(x, x_i) for x_i in self.X_train]
        k_index = np.argsort(_distances)[:self.k]
        k_nearest_labels = []
        k_nearest_distances = []
        for i in k_index:
            k_nearest_labels.append(self.y_train[i])
            k_nearest_distances.append(_distances[i])

    # Tính trọng số dựa trên phương pháp đã chọn
        if self.weight == 'distance':
            weights = []
            for dist in k_nearest_distances:
                if dist == 0:
                    weights.append(1e9)  # Tránh chia cho 0, gán trọng số rất lớn
                else:
                    weights.append(1 / dist)
            # print(print(f"Distance weights: {weights}"))
        
        # Đếm trọng số của từng nhãn
            label_weights = {}
            for idx in range(len(k_nearest_labels)):
                label = k_nearest_labels[idx]
                weight = weights[idx]
                if label in label_weights:
                    label_weights[label] += weight
                else:
                    label_weights[label] = weight
        
        # Lấy nhãn có trọng số lớn nhất
            prediction = max(label_weights, key=label_weights.get)
        else:
        # Nếu weight là 'uniform', tính nhãn phổ biến nhất
            most_common = Counter(k_nearest_labels).most_common(1)
            prediction = most_common[0][0]
        return prediction
  
    





# X_data = result_data[result_data.columns[1:8]]
# y_data = result_data.columns[8:11]

# print(X_data)
# print(y_data)

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
# print(y_data)
# y_train = X_data
# y_test = y_data
# print(X_data)
# print(y_data_tem)
# print(result_data_)
# for i in range(1,100):
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.5, random_state = 42) #random_state=2024
    # print(f"test thu{i}")
    # for j in range(1, 31):

# for i in range (1, 25):

knn = KNN(k=10,weight='distance') #distance
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
total_correct = np.sum(y_test == preds)
accuracy = (total_correct / len(y_test))*100
error = 1 - accuracy
print(f"KNN model's accuracy: {accuracy:.2f}%")

    
# print(f"KNN  model's error: {error:.4f} | Total incorrect predictions: {len(y_test) - total_correct}")
# print(y_test[0:5])
# print(y_train)

# print(f'Perdiction: \t{preds}')
# print(f'y_test: \t{y_test.tolist()}')

#Acurency
# print(y_test == preds)

def predict_student_result(hindi, english, science, maths, history, geography, total):
    # Dữ liệu điểm của học sinh
    student_data = np.array([[hindi, english, science, maths, history, geography, total]])
    
    # Khởi tạo và huấn luyện mô hình
    knn = KNN(k=17)
    knn.fit(X_train, y_train)
    
    # Dự đoán
    prediction = knn.predict(student_data)
    return prediction[0]