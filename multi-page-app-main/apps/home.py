import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Mini Project Data Mining: Prediksi Diabetes", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
   



    with st.sidebar.header('Upload dataset CSV anda'):
        uploaded_file = st.sidebar.file_uploader("Upload file csv anda", type=["csv"])

    k = st.sidebar.slider('K', 1, 25)




    #Main

    st.write("""
    # Data Mining Mini Project
    Dalam project ini sebuah dataset diabetes akan memprediksi outcome / hasil dari beberapa fitur yaitu:
    - *Pregnancies (jumlah kehamilan)*
    - *Glucose (Konsentrasi glukosa)*
    - *BloodPressure (Tekanan Darah / mmHg)*
    - *SkinThickness (ketebalan kulit / mm)*
    - *Insulin (Serum insulin 2 Jam (mu U/ml))*
    - *BMI (Massa Tubuh)*
    - *DiabetesPedigreeFunction (Gen Keturunan Diabetes)*
    - *Age (Umur)*
    - *Outcome (Hasil yang menentukan diabetes atau tidak, **0 berarti tidak diabetes**, **1 berarti diabetes**)*
    """)

    if uploaded_file is not None:
        st.subheader("1. Dataset")
        st.markdown('**Detail dari dataset diabetes**')
        data = pd.read_csv(uploaded_file)
        st.write(data)

        st.subheader("2. Analisis Dataset")
        st.markdown('**2.1 Memeriksa adanya data yang hilang dari setiap fitur**')
        st.write(data.isnull().sum())

        st.markdown('**2.2 Deskripsi Data (count, mean, standar deviasi, min, max, Q1, Q2, Q3)**')
        st.write(data.describe())

        st.markdown("**2.3 Menampilkan distribusi pada kolom 'Outcome'**")
        fig = plt.figure(figsize=(3,4))
        sns.countplot(x = data.Outcome)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        st.markdown("**2.4 Menampilkan korelasi antar atribut**")
        plt.figure(figsize=(8,8))
        sns.clustermap (data.corr(), annot = True, fmt = '.2f')
        
        st.pyplot()

        st.markdown("**2.5 Menampilkan data skewness**")
        plt.figure(figsize=(10,10))
        sns.pairplot(data, hue = 'Outcome',kind = 'scatter',markers='+', diag_kind='kde')
        st.pyplot()

        st.markdown("**2.5 Menghilangkan skewness data dengan menggunakan z-score normalization**")
        y = data['Outcome']
        X = data.drop('Outcome', axis = 1)
        kolom = X.columns
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns = kolom)
        dummy_data = pd.concat([X_scaled,y],axis =1)
        st.write(dummy_data.head())

        st.markdown("**2.5 Membandingkan Outcome dengan variable lainnya**")
        data_melted = pd.melt(dummy_data, id_vars='Outcome')
        st.write(data_melted)

        st.markdown("**2.5 Outlier Detection menggunakan Boxplot**")
        plt.figure(figsize=(20,5))
        sns.boxplot(x=data_melted['variable'], y=data_melted['value'], hue=data_melted['Outcome'])
        st.pyplot()

        st.subheader("3. Data Processing")
        st.markdown('**3.1 Menghilangkan Outlier**')
        st.write('Bentuk Data Sebelum Outlier dihilangkan: ')
        st.info(data.shape)

        for i in data.describe().iloc[:,:-1]:
            Q1 = data.describe().loc['25%', i ]
            Q3 = data.describe().loc['75%', i]
            IQR = Q3-Q1

            Bottom_limit = Q1-1.5*IQR
            Upper_limit = Q3+1.5*IQR

            filter = (data[i] > Bottom_limit) & (data[i] < Upper_limit)
            data = data[filter] 
        st.write('Bentuk Data Sesudah Outlier dihilangkan: ')
        st.info(data.shape)

        y = data['Outcome'] #labels
        X = data.drop('Outcome', axis = 1) #predictor
        X = pd.get_dummies(X)

        #KNN Classification
        st.subheader("4. KNN Classification")


        #membagi data menjadi data training dan data tes 80% dan 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1234)

        #normalisasi
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        
        knn_model = KNeighborsClassifier(n_neighbors = k)
        knn_model.fit(X_train,y_train)

        

        test_pred = knn_model.predict(X_test)

        st.markdown('**4.1 Hasil Klasifikasi**')

        st.markdown('Precision, recall, f1-score, dan support')

        st.info(classification_report(y_test,test_pred))

        st.markdown('Akurasi KNN')
        st.info(accuracy_score(y_test,test_pred))

        st.markdown('Confusion Matrix')

        cnf_matrix = confusion_matrix(y_test, test_pred)

        class_names=[0,1] # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        # create heatmap
        fig = plt.figure(figsize=(4,4))
        
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        ax.xaxis.set_label_position("top")
        
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        buf2 = BytesIO()
        fig.savefig(buf2, format="png")
        st.image(buf2)
        


        st.markdown('Jumlah Prediksi yang benar')
        result_df = pd.DataFrame()
        result_df['y_test'] = np.array(y_test)
        result_df['test_pred'] = test_pred
        result_df['knn_benar'] = result_df['test_pred'] == result_df['y_test']
        fig = plt.figure(figsize=(4,4))
        sns.countplot(x=result_df['knn_benar'], order=[True,False]).set_title('KNN Classification')
        buf3 = BytesIO()
        fig.savefig(buf3, format="png")
        st.image(buf3)

    else:
        st.info('Silahkan upload file CSV anda')
