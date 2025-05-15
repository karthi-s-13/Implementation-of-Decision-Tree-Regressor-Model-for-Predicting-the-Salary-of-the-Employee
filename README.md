# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the unique values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KARTHIKEYAN S
RegisterNumber:  212224230116
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
```
```
data = pd.read_csv('Salary.csv')
df = pd.DataFrame(data)
df.head()
```
```
df['Position'].value_counts().reset_index()
```
```
df['Level'].value_counts().reset_index()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Position'] = le.fit_transform(df['Position'])
df
```
```
x = df.drop(['Salary'], axis=1)
y = df['Salary']
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
```
```
lr = LogisticRegression()
lr.fit(x_train,y_train)
```
```
predict = lr.predict(x_test)
predict
```
```
mse = mean_squared_error(y_test,predict)
print(f"Mean_Square_Error: {mse}")
```
```
r2 = r2_score(y_test,predict)
print(f"R2_Score: {r2}")
```


## Output:
### Dataset
![Screenshot 2025-05-15 084513](https://github.com/user-attachments/assets/c3ffdd84-263c-4ee1-80f2-af23057c9a8f)

### After Encoding
![Screenshot 2025-05-15 084530](https://github.com/user-attachments/assets/ee6d3378-401f-48da-a66c-d435db735d19)

### X value
![Screenshot 2025-05-15 084838](https://github.com/user-attachments/assets/441ce1e8-df60-468d-b902-6c69ba99459c)

### Y value
![Screenshot 2025-05-15 084844](https://github.com/user-attachments/assets/a52f9889-e97c-4725-834b-781803bc0bac)

### Predicted Value
![Screenshot 2025-05-15 084542](https://github.com/user-attachments/assets/15a17cba-9f83-4f9e-a99a-fc5621b67c55)

### Mean Square Error
![Screenshot 2025-05-15 084548](https://github.com/user-attachments/assets/00d79e1d-2167-4fb3-91ad-42536f33d316)

### R^2 Score:
![Screenshot 2025-05-15 084553](https://github.com/user-attachments/assets/f4c76676-afd7-460d-9897-873106e1cef7)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
