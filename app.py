import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

from app_liver_ml import liver_disease_prediction

# ตั้งค่าการแสดงผลให้กว้างขึ้น
st.set_page_config(page_title="การทำนายโรคต่างๆ", layout="wide")

# สร้างหน้าแรกโดยใช้ Sidebar
def app_home():
    # Sidebar สำหรับเลือกหน้า
    page = st.sidebar.selectbox("Choose page", ["Home", "Machine Learning", "Neural Network", "โรคตับ", "โรคเบาหวาน"], key="sidebar_page")

    # ใช้ session_state ในการจัดการการเปลี่ยนหน้า
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # ถ้าเลือกจาก Sidebar ให้กำหนดให้หน้าใน `st.session_state` เป็นค่าที่เลือกจาก Sidebar
    if page != st.session_state.page:
        st.session_state.page = page

    # หน้า "Home"
    if st.session_state.page == "Home":
        # เพิ่มคอลัมน์เพื่อจัดรูปภาพกลางหน้า
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            st.image("image/body.png", caption="Disease Prediction", width=800)  

    # หน้า "Machine Learning"
    elif st.session_state.page == "Machine Learning":
        machine_learning_page()

    # หน้า "Neural Network"
    elif st.session_state.page == "Neural Network":
        neural_network_page()

    # หน้า "โรคตับ"
    elif st.session_state.page == "โรคตับ":
        app_liver_ml()

    # หน้า "โรคเบาหวาน"
    elif st.session_state.page == "โรคเบาหวาน":
        app_diabetes_nn()

# ฟังก์ชันทำนายโรคเบาหวาน
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
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    input_data_scaled = scaler.transform(input_data)

    model.fit(X_train_scaled, y_train)
    prediction = model.predict(input_data_scaled)

    result = "Diabetic (เป็นเบาหวาน)" if prediction == 1 else "Not Diabetic (ไม่เป็นเบาหวาน)"

    results = evaluate_model(model, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    # กราฟแสดงค่าประเมินของโมเดล
    metrics = list(results.values())
    metrics_names = list(results.keys())

    # ลดขนาดกราฟแต่ยังคงความชัดเจน
    fig, ax = plt.subplots(figsize=(5, 2), dpi=150)  

    # สร้างกราฟแบบ bar horizontal
    ax.barh(metrics_names, metrics, color='skyblue')

    # ปรับขนาดตัวอักษรของ x-axis, y-axis และ title ให้เหมาะสม
    ax.set_xlabel('Value', fontsize=8)  
    ax.set_title('Model Evaluation Metrics', fontsize=10)  
    ax.tick_params(axis='y', labelsize=7)  
    ax.tick_params(axis='x', labelsize=7) 

    # แสดงกราฟ
    st.pyplot(fig)

    return result, results

# หน้า "Machine Learning" สำหรับทำนายโรคตับ
def machine_learning_page():
    st.title("การพัฒนา Machine Learning (ML) สำหรับทำนายโรคตับ")

    st.write("""
    โอเค! มาเริ่มกันเลยสำหรับการใช้ *Machine Learning (ML)* เพื่อทำนายว่า *ผู้ป่วยมีโรคตับหรือไม่* กันดีกว่า
    โดยเราจะใช้ข้อมูลของผู้ป่วยเช่น *อายุ*, *เพศ*, *ระดับบิลิรูบิน*, และข้อมูลอื่น ๆ ที่จะช่วยให้เราฝึกโมเดลจนสามารถทำนายได้แบบชัวร์ๆ
    """)

    st.header("1. การพัฒนา Machine Learning (ML)")

    st.write("""
    ในส่วนของ *Machine Learning* นั้น เราจะสร้างโมเดลที่ช่วยในการทำนายโรคตับจากข้อมูลผู้ป่วย มีขั้นตอนหลัก ๆ ดังนี้:
    """)

    # สร้างรายการขั้นตอนการพัฒนา ML
    st.markdown("""
    - **การรวบรวมข้อมูล**: ข้อมูลที่ใช้ฝึกโมเดลมาจาก *Indian Liver Patient Dataset* ซึ่งมีข้อมูลเกี่ยวกับผู้ป่วยโรคตับ เช่น อายุ, เพศ, ระดับบิลิรูบิน ฯลฯ
    - **การเตรียมข้อมูล**: ขั้นตอนนี้เราต้องเติมค่าที่หายไปในข้อมูล (Missing Data) เช่น ใช้ค่าเฉลี่ย (mean) หรือค่าโมด (mode) ในการเติมค่า
    - **การเลือกโมเดล**: โมเดลที่เลือกใช้คือ *Decision Tree* และ *Random Forest*:
        - **Decision Tree**: ใช้ในการแบ่งข้อมูล เช่น ถ้าอายุเกิน 50 ปี ทำนายว่า "มีโรค" หรือ "ไม่มีโรค"
        - **Random Forest**: ใช้หลายๆ *Decision Trees* มาช่วยในการตัดสินใจ ทำให้ได้ผลลัพธ์ที่แม่นยำขึ้น
    - **การฝึกโมเดล (Training)**: ใช้ข้อมูลที่เตรียมไว้ในการฝึกโมเดล เพื่อให้โมเดลสามารถทำนายได้แม่นยำ
    - **การทดสอบโมเดล (Testing)**: ทดสอบโมเดลด้วยข้อมูลที่ไม่เคยใช้ในการฝึก เพื่อประเมินว่าโมเดลทำนายได้แม่นยำแค่ไหน โดยจะใช้ค่าประเมินเช่น *Accuracy*, *Precision*, *Recall*, และ *F1-Score*
    """)

    st.write("ขั้นตอนข้างต้นจะช่วยให้เราเลือกโมเดลที่ดีที่สุดในการทำนายโรคตับ!")

    # เพิ่มข้อมูลตัวอย่างการเลือกโมเดล
    st.write("**ตัวอย่างการเลือกโมเดล**:")
    model_choice = st.selectbox("เลือกโมเดลที่ต้องการใช้:", ["Decision Tree", "Random Forest"])
    
    if model_choice == "Decision Tree":
        st.write("คุณเลือกโมเดล **Decision Tree**")
    elif model_choice == "Random Forest":
        st.write("คุณเลือกโมเดล **Random Forest**")
    
    # ข้อมูลเครดิตของ dataset
    st.markdown("""
    **Dataset:** ข้อมูลที่ใช้ในการฝึกโมเดลนี้มาจาก [Indian Liver Patient Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) จาก Kaggle โดยมีข้อมูลที่เกี่ยวกับผู้ป่วยโรคตับ
    """)

# หน้า "Neural Network" สำหรับทำนายโรคเบาหวาน
def neural_network_page():
    st.title("การพัฒนา Neural Network (NN) สำหรับทำนายโรคเบาหวาน")

    st.write("""
    การพัฒนา *Neural Network (NN)* ใช้ในการทำนายว่า *ผู้ป่วยมีโรคเบาหวานหรือไม่* โดยในที่นี้เราจะใช้ข้อมูลต่างๆ เช่น *จำนวนการตั้งครรภ์*, *ระดับน้ำตาล*, *ความดันโลหิต*, และข้อมูลอื่นๆ ที่ช่วยให้โมเดลเรียนรู้ความสัมพันธ์ได้ดีขึ้น
    """)
    
    st.header("2. การพัฒนา Neural Network (NN) ด้วย MLP")

    st.write("""
    *Multilayer Perceptron (MLP)* คือ Neural Network ที่มีหลายชั้น (layers) ซึ่งใช้ในการเรียนรู้ข้อมูลที่ซับซ้อนได้ดี โดยเราจะใช้ **ReLU** ใน hidden layers และ **Sigmoid** ใน output layer ซึ่งเหมาะสมกับปัญหาการทำนายแบบ **binary classification** เช่น ทำนายว่าโรคเบาหวานหรือไม่
    """)

    st.markdown("""
    - **การรวบรวมข้อมูล**: ข้อมูลที่ใช้ฝึกโมเดล NN คือ *Pima Indians Diabetes Dataset* ซึ่งมีข้อมูลเกี่ยวกับผู้ป่วยโรคเบาหวาน
    - **การเตรียมข้อมูล**: เราจะทำการ *Normalization* หรือ *Scaling* ข้อมูลให้เหมาะสมกับโมเดล เพื่อให้การประมวลผลข้อมูลได้ดีขึ้น
    - **การออกแบบ Neural Network**: 
        - **Hidden Layers**: มี **2 hidden layers** ที่มี **50 neurons** ในแต่ละชั้น
        - **Activation Functions**: ใช้ **ReLU** สำหรับ hidden layers และ **Sigmoid** สำหรับ output layer
    - **การฝึกโมเดล (Training)**: เราจะฝึกโมเดลโดยใช้ข้อมูลที่ทำการสเกลแล้ว และใช้ **Backpropagation** ในการปรับน้ำหนักให้แม่นยำ
    - **การทดสอบโมเดล (Testing)**: เมื่อฝึกเสร็จแล้ว เราจะทดสอบโมเดลกับข้อมูลที่ไม่ได้ใช้ในการฝึก เพื่อประเมินผลลัพธ์ที่ได้จากโมเดล
    """)

    st.write("ขั้นตอนข้างต้นจะช่วยให้เราเลือกโมเดล Neural Network ที่ดีที่สุดในการทำนายโรคเบาหวาน.")

    # เพิ่มข้อมูลตัวอย่างการเลือกโมเดล Neural Network
    st.write("**ตัวอย่างการเลือกโมเดล**:")
    st.write("เราใช้โมเดล **MLP (Multilayer Perceptron)** ซึ่งเหมาะกับการทำนายแบบ binary classification เช่น ทำนายโรคเบาหวาน")

    # สรุปการใช้ MLP
    st.write(""" 
    โมเดล **MLP (Multilayer Perceptron)** ที่ใช้มี **2 hidden layers** ที่มี **50 neurons** ในแต่ละชั้น โดย:
    - **Hidden Layer 1**: 50 neurons, ใช้ **ReLU activation function**
    - **Hidden Layer 2**: 50 neurons, ใช้ **ReLU activation function**
    - **Output Layer**: 1 neuron, ใช้ **Sigmoid activation function** สำหรับการทำนายผลลัพธ์ว่าเป็นโรคเบาหวานหรือไม่
    """)
    st.markdown("""
    **Dataset:** ข้อมูลที่ใช้ในการฝึกโมเดลนี้มาจาก [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) จาก Kaggle ซึ่งเป็นข้อมูลเกี่ยวกับผู้ป่วยโรคเบาหวาน
    """)

# หน้าแสดงการทำนายโรคตับ
def app_liver_ml():
    st.title("การทำนายโรคตับ (Liver Disease Prediction)")  
    
    col1, col2 = st.columns([2, 2]) 

    with col1:
        age = st.slider("Age (อายุ)", 18, 100, 25, key="age_slider")
        gender = st.selectbox("Gender (เพศ)", ['Male', 'Female'], key="gender_selectbox")
        bilirubin = st.slider("Total Bilirubin (ระดับบิลิรูบินรวม)", 0.0, 10.0, 1.0, key="bilirubin_slider")

    with col2:
        model_choice = st.selectbox("เลือกโมเดลที่ต้องการใช้:", ["Decision Tree", "Random Forest"], key="model_choice")

    if st.button("Predict", key="predict_button"):
        # เรียกฟังก์ชัน liver_disease_prediction() ที่ถูกต้อง
        result, results = liver_disease_prediction(age, gender, bilirubin, model_choice)

# แสดงผลลัพธ์ที่ทำนาย
        st.markdown(f"### The predicted result is: **{result}**")
        st.markdown("### Model Evaluation Metrics (ค่าประเมินผลของโมเดล):")

# แสดงค่าประเมินของโมเดล
        for model_name, metrics in results.items():
             st.markdown(f"**{model_name}:**")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                st.markdown(f"- **{metric_name}:** {metric_value:.2f}")
            else:
                st.markdown(f"- **{metric_name}:** {metric_value}")

                    
# หน้าแสดงการทำนายโรคเบาหวาน
def app_diabetes_nn():
    st.title("การทำนายโรคเบาหวานด้วย Neural Network (Diabetes Prediction using Neural Network)") 
    
    col1, col2 = st.columns([1, 1])

    with col1:
        pregnancies = st.number_input("Number of Pregnancies (จำนวนการตั้งครรภ์)", min_value=0, max_value=20, value=1, key="pregnancies")
        glucose = st.number_input("Glucose Level (ระดับน้ำตาลในเลือด)", min_value=0, max_value=200, value=100, key="glucose")
        blood_pressure = st.number_input("Blood Pressure (mm Hg) (ความดันโลหิต, มม. ปรอท)", min_value=0, max_value=200, value=80, key="blood_pressure")
        skin_thickness = st.number_input("Skin Thickness (mm) (ความหนาของผิวหนัง, มม.)", min_value=0, max_value=100, value=20, key="skin_thickness")

    with col2:
        insulin = st.number_input("Insulin Level (ระดับอินซูลิน)", min_value=0, max_value=800, value=100, key="insulin")
        bmi = st.number_input("BMI (ดัชนีมวลกาย)", min_value=10.0, max_value=50.0, value=30.0, key="bmi")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function (ฟังก์ชันสายพันธุ์โรคเบาหวาน)", min_value=0.0, max_value=2.5, value=0.5, key="diabetes_pedigree")
        age = st.number_input("Age (อายุ)", min_value=18, max_value=100, value=25, key="age")

    if st.button("Predict", key="predict_button_diabetes"):
        result, results = diabetes_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
        st.markdown(f"### The predicted result is: **{result}**")
        st.markdown("### Model Evaluation Metrics (ค่าประเมินผลของโมเดล):")
        for metric_name, metric_value in results.items():
            if isinstance(metric_value, (int, float)):
                st.markdown(f"- **{metric_name}:** {metric_value:.2f}")
            else:
                st.markdown(f"- **{metric_name}:** {metric_value}")

# เรียกหน้าแรก
if __name__ == "__main__":
    app_home()
    

