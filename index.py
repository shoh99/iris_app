# import required libraries

from numpy.core.fromnumeric import mean
from scipy.sparse.construct import random
from sklearn import model_selection
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
    'SVM': [],
    'KNN': [],
    'GausianNB': [],
    'LRC': [],
    'DecisionTree': []
}
NAME = ['KNN', 'GausianNB', 'SVM', 'LRC',   'DecisionTree']
RESULTS = []
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
df = pd.read_csv('dataset\iris.csv')
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
            kfold_knn = model_selection.KFold(n_splits=10, random_state=7)
            knn_result = model_selection.cross_val_score(
                knn, X, y, cv=kfold_knn, scoring='accuracy')

            st.write('classifier name: K nearest neighbor algorithm')
            st.write('Accuracy score for your model is: ', knn_result.mean())
            SCORE['KNN'].append(knn_result)
            RESULTS.append(knn_result)

            plt.title('Accuracy score for KNN')
            plt.boxplot(SCORE['KNN'])
            plt.xlabel('KNN')
            st.pyplot()

        # Gausian Naive Baise Algorithm
        if st.checkbox('GausianNB'):
            gnb = GaussianNB()
            # start train
            kfold_gnb = model_selection.KFold(n_splits=10, random_state=7)
            gnb_result = model_selection.cross_val_score(
                gnb, X, y, cv=kfold_gnb, scoring='accuracy'
            )
            SCORE['GausianNB'].append(gnb_result)
            RESULTS.append(gnb_result)
            st.write('classifier name: Gaussian Naive Baise algorithm')
            st.write('Accuracy score for your model is: ', gnb_result.mean())
            # plotbox
            plt.title('Accuracy score for Gaussian Naive Baise')
            plt.boxplot(SCORE['GausianNB'])
            plt.xlabel('GausianNB')
            st.pyplot()

        if st.checkbox('SVC'):
            c = st.slider('C', 0.01, 15.0)
            # g = st.slider('G', 0.01, 15.0)
            # support vector classifier
            svc = SVC(C=c, gamma='scale')

            kfold_svc = model_selection.KFold(n_splits=10, random_state=7)
            svc_result = model_selection.cross_val_score(
                svc, X, y, cv=kfold_svc, scoring='accuracy'
            )
            SCORE['SVM'].append(svc_result)
            RESULTS.append(svc_result)
            st.write('classifier name: Support Vector Machine')
            st.write('Accuracy score for your model is: ', svc_result.mean())

            # plotbox
            plt.title('Accuracy score for Support Vector Classifier')
            plt.boxplot(SCORE['SVM'])
            plt.xlabel('SVM')
            st.pyplot()
        # Logistc Regression
        if st.checkbox('Logistic Regression'):
            lrc = LogisticRegression()

            kfold_lrc = model_selection.KFold(n_splits=10, random_state=7)
            lrc_result = model_selection.cross_val_score(
                lrc, X, y, cv=kfold_lrc, scoring='accuracy'
            )
            SCORE['LRC'].append(lrc_result)
            RESULTS.append(lrc_result)

            st.write('classifier name: Logistic Regresssion')
            st.write('Accuracy score for your model is: ', lrc_result.mean())
            # plotbox
            plt.title('Accuracy score for Logistic Regression')
            plt.boxplot(SCORE['LRC'])
            plt.xlabel('LRC')
            st.pyplot()

        # Decision tree classifier
        if st.checkbox('Decision Tree Classifier'):
            depth = st.slider('max_depth', 1, 15)
            state = st.slider('random_state', 1, 15)

            mad_dt = DecisionTreeClassifier(
                max_depth=depth, random_state=state)

            kfold_dt = model_selection.KFold(n_splits=10, random_state=7)
            dt_result = model_selection.cross_val_score(
                mad_dt, X, y, cv=kfold_dt, scoring='accuracy'
            )
            SCORE['DecisionTree'].append(dt_result)
            RESULTS.append(dt_result)
            st.write('classifier name: Decision Tree Algorithm')
            st.write('Accuracy score for your model is: ', dt_result.mean())

            # plotbox
            plt.title('Accuracy score for Decision Tree')
            plt.boxplot(SCORE['DecisionTree'])
            plt.xlabel('Decision Tree')
            st.pyplot()
    # RESULT
    if st.sidebar.checkbox('Results'):
        st.subheader('Accuracy scores of your models')
        if not SCORE:
            st.write('No model trained')
        else:
            st.write('SVC accuracy score: {} '.format(
                round(SCORE['SVM'][0].mean(), 2)))
            st.write('KNN accuracy score: {} '.format(
                round(SCORE['KNN'][0].mean(), 2)))
            st.write('Gaussian accuracy score: {} '.format(
                round(SCORE['GausianNB'][0].mean(), 2)))
            st.write('Logistic regression accuracy score:{}'.format(
                round(SCORE['LRC'][0].mean(), 2)))
            st.write('Decision Tree Algorithm: {} '.format(
                round(SCORE['DecisionTree'][0].mean(), 2)))

        st.subheader('The overall Boxplot of the result')

        fig = plt.figure()
        fig.suptitle('Algorithm Comparision')
        ax = fig.add_subplot(111)
        plt.boxplot(RESULTS)
        ax.set_xticklabels(NAME)
        st.pyplot()

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
