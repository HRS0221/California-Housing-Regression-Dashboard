# California Housing Regression Dashboard

## Project Overview
This project is an advanced, interactive dashboard for California housing price prediction using linear regression and its variants. It features a Streamlit web app with rich visualizations, model comparison, feature engineering, and explainability.

## Features
- **Multiple Regression Models:** Linear, Polynomial, Ridge, Lasso (with hyperparameters)
- **Feature Engineering:** Region clustering, interaction terms, outlier removal
- **Visualizations:**
  - Geospatial map of predictions/errors
  - Partial dependence plots (PDP)
  - Feature importance (coefficients)
  - Residuals, loss curves, correlation heatmap
- **User Experience:**
  - Downloadable predictions and coefficients
  - Custom data upload for prediction
  - Live model comparison (side-by-side)
  - Modern UI with soft gradient background, card-like containers, and improved footer
  - No copy-link icons on headings for a cleaner look
- **Advanced Analytics:**
  - SHAP explanations for feature impact
  - Error analysis by region and price range
- **Performance & Deployment:**
  - Model persistence (save/load)
  - Streamlit caching for fast, dynamic interactivity
  - Ready for deployment on Streamlit Cloud or Hugging Face Spaces

## Setup
1. **Clone the repository**
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Place `housing.csv` in the project directory.**

## Running the Streamlit App
```bash
streamlit run app.py
```

## Deployment
### Streamlit Cloud
- Push your code to GitHub.
- Go to [Streamlit Cloud](https://share.streamlit.io/) and connect your repo.
- Set the main file as `app.py`.

### Hugging Face Spaces
- Create a new Space, select Streamlit as the SDK.
- Upload your code and `requirements.txt`.
- Set the main file as `app.py`.

## Credits
Project by [Himanshu Salunke](https://github.com/HRS0221/) 