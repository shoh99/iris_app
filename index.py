# import required libraries

import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import plotly_express as px
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from PIL import Image


SCORE = {
    'SVM': None,
    'KNN': None,
    'GausianNB': None,
    'LRC': None,
    'DecisionTree': None
}

# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# set title
st.title('DevOps Team presents')

# import image
image = Image.open('versicolor.jpg')
st.image(image)

# set subtitle
st.write("""
    # A simple ML app with Streamlit
""")

st.write("""
    # Lets Explore different classification with Iris dataset
""")
# function to get dataset
df = pd.read_csv('iris.csv')
st.write(df)


def main():
    df.drop(['Id'], inplace=True, axis=1)

    if st.sidebar.checkbox('EDA'):
        st.subheader('Exploratory data analysis (EDA)')
        # describe dataset
        if st.checkbox('Describe dataset'):

            st.write('Describe the data set')
            st.write(df.describe())

# value count
        if st.checkbox('Display vaules'):

            color_pallete = ['#fc5185', '#3fc1c9', '#364f6b']
            st.write('values of dataset')
            df['Species'].value_counts().plot(kind='bar', color=color_pallete)
            st.pyplot()

# drop the id


# visualization
        if st.checkbox('Visualize data'):

            st.write('Visualize dataset')
            plt.figure(figsize=(8, 8))
            ax = sns.pairplot(df, hue='Species')
            st.pyplot()

# Vusualize 3d plot
        if st.checkbox('Plot 3D plot'):

            fig = px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalLengthCm", size="SepalWidthCm",
                                symbol="Species", color='Species', color_discrete_map={"Joly": "blue", "Bergeron": "violet", "Coderre": "pink"})
            fig.show()
            st.pyplot()

# heat map
        if st.checkbox('Display heat map'):

            plt.figure()
            sns.heatmap(df.corr(), annot=True)
            plt.show()
            st.pyplot()

# train test split

    if st.sidebar.checkbox('Algorithms'):
        st.subheader('Alorithms')
        X = df.drop(['Species'], axis=1)
        y = df['Species']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        # K
        if st.checkbox('KNN'):
            k = st.slider('K', 1, 15)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(X_test)
            st.write(knn_pred)
            accuracy_knn = accuracy_score(y_test, knn_pred)
            st.write('classifier name: Kneighbour algorithm')
            st.write('Accuracy score for your model is: ', accuracy_knn)
            SCORE['KNN'] = accuracy_knn
        # Gausian Naive Baise Algorithm
        if st.checkbox('GausianNB'):
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            gnb_pred = gnb.predict(X_test)
            st.write(gnb_pred)
            accuracy_gnb = accuracy_score(y_test, gnb_pred)
            st.write('classifier name: Gaussian Naive BAise algorithm')
            st.write('Accuracy score for your model is: ', accuracy_gnb)
            SCORE['GausianNB'] = accuracy_gnb

        if st.checkbox('SVC'):
            c = st.slider('C', 0.01, 15.0)
            g = st.slider('G', 0.01, 15.0)
            # support vector classifier
            svc = SVC(C=c, gamma=g)
            svc.fit(X_train, y_train)
            svc_pred = svc.predict(X_test)
            st.write(svc_pred)
            accuracy_svc = accuracy_score(y_test, svc_pred)
            st.write('classifier name: Support Vector Machine')
            st.write('Accuracy score for your model is: ', accuracy_svc)
            SCORE['SVM'] = accuracy_svc
        # Logistc Regression
        if st.checkbox('Logistic Regression'):
            lrc = LogisticRegression()
            lrc.fit(X_train, y_train)
            pred = lrc.predict(X_test)
            st.write(pred)
            accuracy_lrc = accuracy_score(pred, y_test)
            st.write('classifier name: Logistic Regresssion')
            st.write('Accuracy score for your model is: ', accuracy_lrc)
            SCORE['LRC'] = accuracy_lrc
        # Decision tree classifier
        if st.checkbox('Decision Tree Classifier'):
            depth = st.slider('max_depth', 1, 15)
            state = st.slider('random_state', 1, 15)

            mad_dt = DecisionTreeClassifier(
                max_depth=depth, random_state=state)
            mad_dt.fit(X_train, y_train)
            mad_dt_pred = mad_dt.predict(X_test)
            st.write(mad_dt_pred)
            accuracy_mad_dt = accuracy_score(y_test, mad_dt_pred)
            st.write('classifier name: Decision Tree Algorithm')
            st.write('Accuracy score for your model is: ', accuracy_mad_dt)
            SCORE['DecisionTree'] = accuracy_mad_dt

    # RESULT
    if st.sidebar.checkbox('Results'):
        st.subheader('Accuracy scores of your models')
        if not SCORE:
            st.write('No model trained')
        else:
            st.write('SVC accuracy score: ', SCORE['SVM'])
            st.write('KNN accuracy score: ', SCORE['KNN'])
            st.write('Gaussian accuracy score: ', SCORE['GausianNB'])
            st.write('Logistic regression accuracy score: ', SCORE['LRC'])
            st.write('Decision Tree Algorithm: ', SCORE['DecisionTree'])

    if st.sidebar.checkbox('About'):
        st.subheader('About this app')
        st.write("""
            The aim is to classify iris flowers among three species (setosa, versicolor, or virginica) from measurements of sepals and petals' length and width.

            The iris data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

            This app uses 5 classification algorithms and check accuracy score between each others
        """)
        st.balloons()


if __name__ == "__main__":
    main()
