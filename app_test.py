import streamlit as st
from streamlit.testing.v1 import AppTest
import numpy as np

def test_app_loads():
    at = AppTest.from_file("app.py")
    at.run()
    # Check that the sidebar loads and has the expected header
    assert at.sidebar.header[0].value == "Feature Selection"
    # Check that the main page loads
    assert at.title[0].value == "California Housing Regression Dashboard"

def test_feature_selection_and_model_training():
    at = AppTest.from_file("app.py")
    at.run()
    # Select features in the sidebar
    at.sidebar.multiselect[0].select("median_income")
    # Select model type
    at.sidebar.selectbox[0].select("Linear")
    # Trigger rerun
    at.run()
    # Check that metrics are displayed
    assert any("MSE" in el.value for el in at.markdown)
    assert any("R²" in el.value for el in at.markdown)

def test_train_and_predict():
    at = AppTest.from_file("app.py")
    at.run()
    # Select features and model
    at.sidebar.multiselect[0].select("median_income")
    at.sidebar.selectbox[0].select("Linear")
    at.run()
    # Check that the model is trained and predictions are made
    assert any("Prediction" in el.value for el in at.markdown) 