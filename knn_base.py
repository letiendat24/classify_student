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
            weight (str): use to chose 2 options (uniforms and distance) for weight calculation
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
                preds(np.ndarray): list of predicted label
        """
        # preds = []
        # for x in enumerate(X_test):
        return np.array([self._predict_one(x) for x in X_test])
        
        
    def _euclidean_distance(self, p, q):
        return np.sqrt(np.sum((p - q) ** 2))

    def _manhattan_distance(self, p, q):
        return np.sum(np.abs(p - q))
    
   


#tinh kc cua x voi cac phantu x train
    def _predict_one(self, x:np.ndarray):
        """
            Handle sorting and choosing labelslabels
            Args: 
                x: a tube in X_test
            
            Return:
                predictation(int): predicted label
        
        """  
        _distances =  np.array([self._euclidean_distance(x, x_i) for x_i in self.X_train])
        
        #lay ra chi so cua K diem gan nhat
        k_index = np.argsort(_distances)[:self.k]

        #lay ra K labels tuong ung
        k_nearest_labels = self.y_train[k_index]
        
        #lay ra K khoang cach gan nhat
        k_nearest_distances = _distances[k_index]

        #tinh trong so
        if self.weight == "distance":
        #tao mang luu trong so cho tung nhan
            unique_labels = np.unique(k_nearest_labels)
            label_weights = np.zeros(unique_labels.shape)

            for i , label in enumerate(unique_labels):
                label_filter = (label == k_nearest_labels)
                label_distance = k_nearest_distances[label_filter]
                
                weights = np.sum(1 /label_distance)
                label_weights[i] = weights
            #tim label co trong so max
            max_weight_index = np.argmax(label_weights)
            predictation = unique_labels[max_weight_index]
        else:
            most_common = Counter(k_nearest_labels).most_common()
            predictation = most_common[0][0]
        return predictation
        
        
    
f = open('D:\\python\\lab\\project\\results.csv', 'r')
result_data = csv.reader(f)
result_data_str = np.array(list(result_data))
result_data_str = np.delete(result_data_str, 0, 0) 
result_data_str = np.delete(result_data_str, 0, 1) 

result_data_ = result_data_str.astype(int)



X_data = np.delete(result_data_,[7,8], 1)
y_data = np.delete(result_data_,[0,1,2,3,4,5,6,7], 1).flatten()


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42) #random_state=2024


for i in range (1, 25):

    knn = KNN(k=i, weight='uniforms') #distance uniforms
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    total_correct = np.sum(y_test == preds)
    accuracy = (total_correct / len(y_test))*100
    error = 1 - accuracy
    print(f"KNN {i} model's accuracy: {accuracy:.2f}%")
    

    
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