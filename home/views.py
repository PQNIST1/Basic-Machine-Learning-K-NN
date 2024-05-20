from django.shortcuts import render
import pandas as pd
from home.module import classify_customer, euclidean_distance, predict_spending_score
import numpy as np
def home(request):
    c_predict = ''
    d_predict = 0
    similar_customers_info = []
    if request.method == 'POST':
        data = pd.read_csv('d:\Python2\code\Machine_Django\Mall_Customers (1).csv')
        data.head()
        df1=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
        X=df1[["Annual Income (k$)","Spending Score (1-100)"]]
        # Chọn các cột 'Age' và 'Annual Income (k$)' làm đặc trưng (features)
        X_train = data[['Age', 'Annual Income (k$)']].values
        # Lấy cột 'Spending Score (1-100)' làm biến mục tiêu
        y_train = data['Spending Score (1-100)'].values
        genders = data['Gender'].values
        age = int(request.POST['age'])
        salary = int(request.POST['salary'])
        k = int(request.POST['customer'])
        gender = request.POST['gender']
        x_test = np.array([age, salary])
        predicted_spending_score = predict_spending_score(X_train, y_train, x_test, k)
        similar_customers_indices = np.argsort([euclidean_distance(x_test, x) for x in X_train])[:k]
        classification = classify_customer(x_test,predicted_spending_score ,gender)
        c_predict = classification
        d_predict = predicted_spending_score
        for idx in similar_customers_indices:
            info = {
                'age': X_train[idx][0],
                'income': X_train[idx][1],
                'spending_score': y_train[idx],
                'gender': genders[idx]
            }
            similar_customers_info.append(info)
        return render(request, 'pages/home.html',{"c_predict":c_predict,"d_predict":d_predict, "similar_customers_info":similar_customers_info})
    return render(request, 'pages/home.html',{"c_predict":c_predict,"d_predict":d_predict, "similar_customers_info":similar_customers_info})
# Create your views here.
