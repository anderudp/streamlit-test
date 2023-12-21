import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import uniform
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, make_scorer


if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    
    # Quick rename
    df.rename(columns={"DiabetesPedigreeFunction":"DPF"}, inplace=True)
    
    # Drop 3, 4 invalid records
    df["invalids"] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isin([0, 0.0, np.nan]).sum(axis=1)
    df.drop(df[df["invalids"] >= 3].index, inplace=True)
    df.drop(["invalids"], axis=1, inplace=True)
    
    # Replace zeroes with column avg on affected datapoints
    df['Glucose'].replace(0, np.NaN, inplace=True)
    df['BloodPressure'].replace(0, np.NaN, inplace=True)
    df['SkinThickness'].replace(0, np.NaN, inplace=True)
    df['Insulin'].replace(0, np.NaN, inplace=True)
    df['BMI'].replace(0, np.NaN, inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    # Split that bih
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Logreg + optimize
    model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)
    param_grid = {
        'C': [0.01],
        'penalty': ['l2']
    }
    grid_search = GridSearchCV(model, param_grid, scoring='recall', cv=100)
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Use the best parameters to create the final logistic regression model
    final_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], max_iter=1000)

    # Train the final model on the entire training set
    final_model.fit(X_train, y_train)

    # Frontend
    st.title("Cukorbetegség predikció")
    pregnancies = st.number_input("Terhességek száma (#)", value=round(df['Pregnancies'].mean()), min_value=0)
    glucose = st.number_input("OGTT-2 vérglükóz (mg/dL)", value=round(df['Glucose'].mean()), min_value=0)
    blood_pressure = st.number_input("Diasztolés vérnyomás (mmHg)", value=round(df['BloodPressure'].mean()), min_value=0)
    skin_thickness = st.number_input("Tricepsz bőrredővastagsága (mm)", value=round(df['SkinThickness'].mean()), min_value=0)
    insulin = st.number_input("Biokatív vérinzulin (μU/mL)", value=round(df['Insulin'].mean()), min_value=0)
    bmi = st.number_input("Testtömegindex (kg/m)", value=round(df['BMI'].mean(), 1), min_value=0.0)
    dpf = st.number_input("Származás-alapú diabéteszfüggvény (DPF)", value=round(df['DPF'].mean(), 3), min_value=0.0)
    age = st.number_input("Életkor (év)", value=round(df['Age'].mean()), min_value=0)
    result = None
    def btn_callback():
        result = final_model.predict(np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]))
        result_text = "Várhatóan pozitív" if result[0] == 1 else "Várhatóan negatív"
        st.title(f"Eredmény: {result_text}")

    st.button("Predikció indítása", on_click=btn_callback)
    