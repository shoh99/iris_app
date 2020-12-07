# import required libraries

import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import plotly_express as px
from mpl_toolkits.mplot3d import Axes3D
import time

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

TIME = {
    'SVM_time': [0.0, 0.0],
    'KNN_time': [0.0, 0.0],
    'GausianNB_time': [0.0, 0.0],
    'LRC_time': [0.0, 0.0],
    'DecisionTree_time': [0.0, 0.0]
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
df = pd.read_csv('Iris.csv')
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
            start_train = time.time()
            knn.fit(X_train, y_train)
            end_train = time.time()
            start_test = time.time()
            knn_pred = knn.predict(X_test)
            end_test = time.time()
            st.write(knn_pred)
            accuracy_knn = accuracy_score(y_test, knn_pred)
            st.write('classifier name: K nearest neighbor algorithm')
            st.write('Accuracy score for your model is: ', accuracy_knn)
            st.write(
                f"Train_time: {end_train-start_train}, Test_time: {end_test-start_test}")
            SCORE['KNN'] = accuracy_knn
            TIME['KNN_time'] = [end_train-start_train, end_test-start_test]
        # Gausian Naive Baise Algorithm
        if st.checkbox('GausianNB'):
            gnb = GaussianNB()
            # start train
            gnb_train = time.time()
            gnb.fit(X_train, y_train)
            gnb_train_end = time.time()
            # start test
            gnb_test = time.time()
            gnb_pred = gnb.predict(X_test)
            gnb_test_end = time.time()
            st.write(gnb_pred)
            accuracy_gnb = accuracy_score(y_test, gnb_pred)
            st.write('classifier name: Gaussian Naive Baise algorithm')
            st.write('Accuracy score for your model is: ', accuracy_gnb)
            st.write(
                f"Train_time: {gnb_test_end-gnb_train}, Test_time: {gnb_test_end-gnb_test}")
            SCORE['GausianNB'] = accuracy_gnb
            TIME['GausianNB_time'] = [
                gnb_test_end-gnb_train, gnb_test_end-gnb_test]

        if st.checkbox('SVC'):
            c = st.slider('C', 0.01, 15.0)
            g = st.slider('G', 0.01, 15.0)
            # support vector classifier
            svc = SVC(C=c, gamma=g)
            # start train
            svc_train = time.time()
            svc.fit(X_train, y_train)
            svc_train_end = time.time()
            # start test
            svc_test = time.time()
            svc_pred = svc.predict(X_test)
            # end_test
            svc_test_end = time.time()
            st.write(svc_pred)
            accuracy_svc = accuracy_score(y_test, svc_pred)
            st.write('classifier name: Support Vector Machine')
            st.write('Accuracy score for your model is: ', accuracy_svc)
            st.write(
                f"Train_time: {svc_train_end-svc_train}, Test_time: {svc_test_end-svc_test}")
            SCORE['SVM'] = accuracy_svc
            TIME['SVM_time'] = [svc_train_end-svc_train, svc_test_end-svc_test]
        # Logistc Regression
        if st.checkbox('Logistic Regression'):
            lrc = LogisticRegression()
            # start train
            lrc_train = time.time()
            lrc.fit(X_train, y_train)
            # end train
            lrc_train_end = time.time()
            # start test
            lrc_test = time.time()
            pred = lrc.predict(X_test)
            # end test
            lrc_test_end = time.time()
            st.write(pred)
            accuracy_lrc = accuracy_score(pred, y_test)
            st.write('classifier name: Logistic Regresssion')
            st.write('Accuracy score for your model is: ', accuracy_lrc)
            st.write(
                f"Train_time: {lrc_train_end-lrc_train}, Test_time: {lrc_test_end-lrc_test}")
            SCORE['LRC'] = accuracy_lrc
            TIME['LRC_time'] = [lrc_train_end-lrc_train, lrc_test_end-lrc_test]
        # Decision tree classifier
        if st.checkbox('Decision Tree Classifier'):
            depth = st.slider('max_depth', 1, 15)
            state = st.slider('random_state', 1, 15)

            mad_dt = DecisionTreeClassifier(
                max_depth=depth, random_state=state)
            # start train
            mad_dt_train = time.time()
            mad_dt.fit(X_train, y_train)
            # end train
            mad_dt_train_end = time.time()
            # start test
            mad_dt_test = time.time()
            mad_dt_pred = mad_dt.predict(X_test)
            # end test
            mad_dt_test_end = time.time()
            st.write(mad_dt_pred)
            accuracy_mad_dt = accuracy_score(y_test, mad_dt_pred)
            st.write('classifier name: Decision Tree Algorithm')
            st.write('Accuracy score for your model is: ', accuracy_mad_dt)
            st.write(
                f"Train_time: {mad_dt_train_end-mad_dt_train}, Test_time: {mad_dt_test_end-mad_dt_test}")
            SCORE['DecisionTree'] = accuracy_mad_dt
            TIME['DecisionTree_time'] = [mad_dt_train_end -
                                         mad_dt_train, mad_dt_test_end-mad_dt_test]

    # RESULT
    if st.sidebar.checkbox('Results'):
        st.subheader('Accuracy scores of your models')
        if not SCORE:
            st.write('No model trained')
        else:
            st.write('SVC accuracy score:{}, Train time: {}, Test time: {} '.format(
                SCORE['SVM'], TIME['SVM_time'][0], TIME['SVM_time'][1]))
            st.write('KNN accuracy score: {}, Train time: {}, Test time: {}'.format(
                SCORE['KNN'], TIME['KNN_time'][0], TIME['KNN_time'][1]))
            st.write('Gaussian accuracy score:{}, Train time: {}, Test time: {} '.format(
                SCORE['GausianNB'], TIME['GausianNB_time'][0], TIME['GausianNB_time'][1]))
            st.write('Logistic regression accuracy score:{}, Train time: {}, Test time: {} '.format(
                SCORE['LRC'], TIME['LRC_time'][0], TIME['LRC_time'][1]))
            st.write('Decision Tree Algorithm:{}, Train time: {}, Test time: {} '.format(
                SCORE['DecisionTree'], TIME['DecisionTree_time'][0], TIME['DecisionTree_time'][1]))

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
