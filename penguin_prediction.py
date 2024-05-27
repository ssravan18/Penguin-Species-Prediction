import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
path = "S:\\Whatsapp\\projects\\penguin\\penguins_size.csv"
df = pd.read_csv(path)

# Title and description
st.title("Penguin Species Classification")
st.write("""
This app uses a K-Nearest Neighbors (KNN) classifier to predict the species of penguins based on their physical measurements.
""")

# Display the dataframe
st.subheader("Data")
st.dataframe(df.head())

# EDA (Exploratory Data Analysis)
st.subheader("Exploratory Data Analysis")

# Countplot of species
st.write("Countplot of species:")
fig1, ax1 = plt.subplots()
sns.countplot(x=df['species'], ax=ax1)
st.pyplot(fig1)

# Countplot of species by sex
st.write("Countplot of species by sex:")
fig2, ax2 = plt.subplots()
sns.countplot(x=df['species'], hue=df['sex'], ax=ax2)
st.pyplot(fig2)

# Jointplot of culmen length and depth
st.write("Jointplot of culmen length and depth:")
fig3 = sns.jointplot(x="culmen_length_mm", y="culmen_depth_mm", data=df)
st.pyplot(fig3)

# Scatter plot of culmen length and depth by species
st.write("Scatter plot of culmen length and depth by species:")
fig4, ax4 = plt.subplots()
sns.scatterplot(x="culmen_length_mm", y="culmen_depth_mm", data=df, hue="species", ax=ax4)
st.pyplot(fig4)

# Countplot of species by island
st.write("Countplot of species by island:")
fig5, ax5 = plt.subplots()
sns.countplot(x=df['species'], hue=df['island'], ax=ax5)
st.pyplot(fig5)

# Data Preprocessing
st.subheader("Data Preprocessing")

# Fill missing values with 0
df = df.fillna(0)

# Converting Categorical Data to Numerical
df = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)

# Ensure all columns are included during prediction
required_columns = df.drop('species', axis=1).columns.tolist()

# Display the dataframe after transformation
st.write("Data after converting categorical variables to numerical:")
st.dataframe(df.head())

# Check for missing values
st.write("Heatmap of missing values:")
fig6, ax6 = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax6)
st.pyplot(fig6)

# Standardizing the data
scale = StandardScaler()
X = df.drop(['species'], axis=1)
Y = df['species']
scale.fit(X)
transformed = scale.transform(X)
df_scaled = pd.DataFrame(transformed, columns=X.columns)
st.write("Data after standardization:")
st.dataframe(df_scaled.head())

# Build and evaluate the classification model
st.subheader("Model Building and Evaluation")

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_scaled, Y, test_size=0.33, random_state=101)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Predict the labels for the test set
output = knn.predict(x_test)

# Display the confusion matrix
st.write("Confusion matrix:")
cm = confusion_matrix(y_test, output)
st.write(cm)

# User input for prediction
st.subheader("Make a Prediction")

def user_input_features():
    culmen_length_mm = st.sidebar.slider('Culmen Length (mm)', float(df['culmen_length_mm'].min()), float(df['culmen_length_mm'].max()))
    culmen_depth_mm = st.sidebar.slider('Culmen Depth (mm)', float(df['culmen_depth_mm'].min()), float(df['culmen_depth_mm'].max()))
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
    body_mass_g = st.sidebar.slider('Body Mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    island = st.sidebar.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])

    sex_Male = 1 if sex == 'Male' else 0
    island_Dream = 1 if island == 'Dream' else 0
    island_Torgersen = 1 if island == 'Torgersen' else 0

    data = {'culmen_length_mm': culmen_length_mm,
            'culmen_depth_mm': culmen_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex_Male': sex_Male,
            'island_Dream': island_Dream,
            'island_Torgersen': island_Torgersen}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Add a button to make prediction
if st.button('Predict'):
    # Ensure input_df has all the required columns
    input_df = input_df.reindex(columns=required_columns, fill_value=0)

    # Standardize user input
    input_scaled = scale.transform(input_df)

    # Make prediction
    prediction = knn.predict(input_scaled)

    # Display prediction
    st.subheader('Prediction')
    st.markdown(f"<h2 style='font-size: 24px;'>The predicted species is: {prediction[0]}</h2>", unsafe_allow_html=True)
