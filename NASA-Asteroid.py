import pandas as pd #Data manipulation
import numpy as np #Numerical computations
from sklearn.neighbors import KNeighborsClassifier #KNN Classifier
from sklearn.model_selection import train_test_split #Train-Test Split
from sklearn.metrics import accuracy_score #Accuracy Metric
from sklearn.preprocessing import StandardScaler #Feature Scaling
import matplotlib.pyplot as plt #Plotting
import time #Time Measurement

#Data Loading and Preprocessing
df=pd.read_csv("neo_v2.csv")
df.fillna(0,inplace=True)
df=df.drop_duplicates(subset='name',keep='first')
X=np.array(df.loc[:,["est_diameter_min","est_diameter_max", "miss_distance","relative_velocity","absolute_magnitude"]])
X=StandardScaler().fit_transform(X)
Y=np.array(df.iloc[:,-1])

accuracy_list=[]
start=time.time()
max=0
best_k=0

# Finding the best K value for KNN Classifier
for k in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=k)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=67)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    accuracy_list.append(accuracy_score(y_test,y_pred)*100)
    
    if accuracy_list[-1]>max:
        max=accuracy_list[-1]
        best_k=k
        
end=time.time()

# Plotting the accuracy vs Number of neighbors(K)  
plt.plot(range(1,100),accuracy_list,color="#ed4f24")
plt.title("KNN Classifier Accuracy vs Number of Neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuacracy (%)")  
plt.show()

# Displaying the best accuracy and corresponding K value
print(f"Best Accuracy: {max}% at K={best_k}")
print(f"Time taken: {end-start:.2f} seconds")
