import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import seaborn as sns

def app():
    with st.sidebar.header('Upload dataset CSV anda'):
        uploaded_file = st.sidebar.file_uploader("Upload file csv anda", type=["csv"])

    k = st.sidebar.slider('K', 1, 25)

    st.title('Prediksi dengan KNN')

    



    


    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.markdown('**Detail dari dataset diabetes**')
        
        st.write(data)

        for i in data.describe().iloc[:,:-1]:
            Q1 = data.describe().loc['25%', i ]
            Q3 = data.describe().loc['75%', i]
            IQR = Q3-Q1

            Bottom_limit = Q1-1.5*IQR
            Upper_limit = Q3+1.5*IQR

            filter = (data[i] > Bottom_limit) & (data[i] < Upper_limit)
            data = data[filter] 
    
        y = data['Outcome'] #labels
        X = data.drop('Outcome', axis = 1) #predictor
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1234)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        
        knn_model = KNeighborsClassifier(n_neighbors = k)
        knn_model.fit(X_train,y_train)

        test_pred = knn_model.predict(X_test)

        with st.form(key='my_form_to_submit'):

            st.markdown("**Masukan Parameter yang Ingin Diprediksi**")

            preg = st.number_input('Pregnancies / Jumlah Kehamilan', 0,100,step=1)

            
            Glucose = st.number_input('Glucose / Konsetrasi Glukosa',1,200, step=1)

            
            bp = st.number_input('Blood Pressure / Tekanana darah diastole', 1,200,step=1)

            
            skin = st.number_input('Skin Thickness / ketebalan kulit (mm)',1,50, step=1)

            
            insulin = st.number_input('Insulin (mu U/ml)', 0,1000,step=1)

            
            bmi = st.number_input('BMI (Index Massa Tubuh)',00.0000)

            
            dpf = st.number_input('DiabetesPedigreeFunction (Gen Keturunan Diabetes)', 0.000)
            dpf = dpf/100
            age = st.number_input('Age / Umur', 1,100,step=1)

            # inputdata = np.array([[]])

            knn_model.fit(X,y)

            predc = {0:'Kemungkinan Tidak Diabetes', 1:'Kemungkinan Diabetes'}

            prediction = knn_model.predict([[preg,Glucose,bp,skin,insulin,bmi,dpf,age]])

            

    
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.info(predc[prediction[0]])
            
            


        

        

    else:
        st.info('Silahkan upload file CSV anda')
