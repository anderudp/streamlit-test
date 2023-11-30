import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()
api.dataset_download_files('uciml/pima-indians-diabetes-database', unzip=True)

st.title("Pima indians diabetes regression")
