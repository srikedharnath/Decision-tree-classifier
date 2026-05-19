import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
)

y = iris.target


model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)


st.title("🌸 Iris Flower Prediction using Decision Tree")

st.write("Enter the flower measurements below:")


sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)


if st.button("Predict Flower"):
    
    input_data = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    flower_name = iris.target_names[prediction]

    st.success(f"Predicted Flower: {flower_name}")