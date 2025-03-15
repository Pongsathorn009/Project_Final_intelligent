import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

def diabetes_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    dataset = pd.read_csv('pima_indians_diabetes.csv')

    # เติมค่าที่ขาดหายด้วยค่ากลาง (mean) สำหรับฟีเจอร์ที่เป็นตัวเลข
    imputer = SimpleImputer(strategy='mean')
    dataset.iloc[:, :] = imputer.fit_transform(dataset)

    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']

    model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def evaluate_model(model, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)

        return {
            'Accuracy (ความแม่นยำ)': accuracy,
            'Precision (ความแม่นยำเฉพาะ)': precision,
            'Recall (การเรียกคืน)': recall,
            'F1-Score (คะแนน F1)': f1
        }

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    input_data_scaled = scaler.transform(input_data)

    model.fit(X_train_scaled, y_train)
    prediction = model.predict(input_data_scaled)

    result = "Diabetic (เป็นเบาหวาน)" if prediction == 1 else "Not Diabetic (ไม่เป็นเบาหวาน)"

    results = evaluate_model(model, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    return result, results
