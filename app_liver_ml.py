# app_liver_ml.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def liver_disease_prediction(age, gender, bilirubin, model_choice):
    # อ่านข้อมูลจากไฟล์ CSV
    dataset = pd.read_csv('indian_liver_patient.csv')

    # แปลงคอลัมน์ 'Gender' เป็นตัวเลข
    label_encoder = LabelEncoder()
    dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])

    # เติมค่าที่ขาดหายด้วยค่ากลาง (mean) สำหรับฟีเจอร์ที่เป็นตัวเลข
    imputer = SimpleImputer(strategy='mean')
    dataset.iloc[:, :] = imputer.fit_transform(dataset)

    # แยก features และ target
    X = dataset.drop('Dataset', axis=1)
    y = dataset['Dataset']
    y = y - 1

    # สร้างโมเดล
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ใช้ Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ฟังก์ชันในการฝึกและทำนาย
    def evaluate_model(models, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
        results = {}

        for name, model in models.items():
            if name == model_choice:
                if name == 'Random Forest':
                    model.fit(X_train, y_train)  # ใช้ข้อมูลที่ไม่ได้สเกล
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)  # ใช้ข้อมูลที่สเกล
                    y_pred = model.predict(X_test_scaled)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=1)
                recall = recall_score(y_test, y_pred, zero_division=1)
                f1 = f1_score(y_test, y_pred, zero_division=1)

                results[name] = {
                    'Accuracy (ความแม่นยำ)': accuracy,
                    'Precision (ความแม่นยำเฉพาะ)': precision,
                    'Recall (การเรียกคืน)': recall,
                    'F1-Score (คะแนน F1)': f1
                }

        return results

    # ข้อมูลที่ใช้ในการทำนาย
    input_data = np.array([[age, 1 if gender == 'Male' else 0, bilirubin, 0, 0, 0, 0, 0, 0, 0]])

    # ทำการสเกลข้อมูลที่กรอกให้เหมือนข้อมูลที่ใช้ฝึก
    input_data_scaled = scaler.transform(input_data)

    # ทำนายผล
    model = models[model_choice]

    if model_choice == 'Random Forest':
        model.fit(X_train, y_train)
        prediction = model.predict(input_data)
    else:
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(input_data_scaled)

    result = "Liver Disease (โรคตับ)" if prediction == 1 else "No Liver Disease (ไม่มีโรคตับ)"

    # ประเมินผล
    results = evaluate_model(models, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    return result, results