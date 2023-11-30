import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd


api = KaggleApi()
api.authenticate()
api.dataset_download_files('uciml/pima-indians-diabetes-database', unzip=True)

df = pd.read_csv('diabetes.csv')

st.title("Pima indians diabetes regression")
st.selectbox("What field should we predict for?", df.columns.sort_values())