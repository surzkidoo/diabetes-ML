from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

@api_view(['GET', 'POST'])
def logistic(request):
    data = pd.read_csv(r"C:\Users\surzkid\Desktop\diabetes.csv")
    x=data.drop("Outcome", axis=1)
    y=data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model= LogisticRegression()
    model.fit(x_train,y_train)
    val1=float(request.data[0])
    val2=float(request.data[1])
    val3=float(request.data[2])
    val4=float(request.data[3])
    val5=float(request.data[4])
    val6=float(request.data[5])
    val7=float(request.data[6])
    val8=float(request.data[7])
    result=model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    return Response(result)
@api_view(['GET', 'POST'])

def random(request):
    data = pd.read_csv(r"C:\Users\surzkid\Desktop\diabetes.csv")
    x=data.drop("Outcome", axis=1)
    y=data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model= RandomForestClassifier(n_estimators=100)
    model.fit(x_train,y_train)
    val1=float(request.data[0])
    val2=float(request.data[1])
    val3=float(request.data[2])
    val4=float(request.data[3])
    val5=float(request.data[4])
    val6=float(request.data[5])
    val7=float(request.data[6])
    val8=float(request.data[7])
    result=model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    return Response(result)

@api_view(['GET', 'POST'])
def decision(request):
    data = pd.read_csv(r"C:\Users\surzkid\Desktop\diabetes.csv")
    x=data.drop("Outcome", axis=1)
    y=data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)
    val1=float(request.data[0])
    val2=float(request.data[1])
    val3=float(request.data[2])
    val4=float(request.data[3])
    val5=float(request.data[4])
    val6=float(request.data[5])
    val7=float(request.data[6])
    val8=float(request.data[7])
    result=model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    return Response(result)