import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
from sklearn.inspection import PartialDependenceDisplay
import io
import shap
import joblib

st.set_page_config(page_title="California Housing Regression Dashboard", layout="centered")

# --- Caching for performance ---
@st.cache_data
def load_data():
    data = pd.read_csv('housing.csv')
    data = data.dropna(subset=['total_bedrooms'])
    # Always assign Region column for error analysis and plotting
    if 'Region' not in data.columns:
        def assign_region(row):
            return 'Coastal' if row['longitude'] < -120 else 'Inland'
        data['Region'] = data.apply(assign_region, axis=1)
    # Feature engineering
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

data = load_data()

# Feature selection
all_features = [
    'median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

st.sidebar.header("Feature Selection")
selected_features = st.sidebar.multiselect(
    "Select features for regression:", all_features, default=['median_income']
)

st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Select regression model:",
    ["Linear", "Polynomial", "Ridge", "Lasso"]
)

if model_type == "Polynomial":
    poly_degree = st.sidebar.slider("Polynomial Degree", min_value=2, max_value=5, value=2, step=1)
else:
    poly_degree = 1

if model_type in ["Ridge", "Lasso"]:
    reg_alpha = st.sidebar.slider("Regularization Strength (alpha)", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
else:
    reg_alpha = 1.0  # Ensure always a float for linter

st.sidebar.header("Data Split & Validation")
test_size = st.sidebar.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
cv_folds = st.sidebar.slider("K-Fold Cross-Validation (k)", min_value=2, max_value=10, value=5, step=1)

# Prepare X and y
X = data[selected_features].to_numpy()
y = data['median_house_value'].to_numpy().reshape(-1, 1)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Polynomial features
poly = None   # Ensure poly is always defined
if poly_degree > 1:
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)
    feature_names = poly.get_feature_names_out(selected_features)
else:
    X_poly = X_norm
    feature_names = selected_features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=42)

# Model selection
model = None  # Ensure model is always defined
if model_type == "Linear":
    model = LinearRegression()
elif model_type == "Polynomial":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=reg_alpha)
elif model_type == "Lasso":
    model = Lasso(alpha=reg_alpha, max_iter=10000)
else:
    st.error("Unknown model type selected.")
    st.stop()  # Stop Streamlit execution if model type is invalid

model.fit(X_train, y_train)

# If model is not None, predict; otherwise, set predictions to zeros
if model is not None:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
else:
    st.error("Model is not defined. Cannot predict train/test values.")
    y_pred_train = np.zeros_like(y_train)
    y_pred_test = np.zeros_like(y_test)

# Cross-validation
cv_scores = cross_val_score(model, X_poly, y, cv=cv_folds, scoring='r2')

# Evaluation metrics
def get_metrics(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    min_len = min(len(y_true), len(y_pred))
    mse = mean_squared_error(y_true[:min_len], y_pred[:min_len])
    mae = mean_absolute_error(y_true[:min_len], y_pred[:min_len])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true[:min_len], y_pred[:min_len])
    return mse, mae, rmse, r2

mse_train, mae_train, rmse_train, r2_train = get_metrics(y_train, y_pred_train)
mse_test, mae_test, rmse_test, r2_test = get_metrics(y_test, y_pred_test)

# --- Precompute all figures and DataFrames for dashboard tabs ---
# Plot predictions (for single feature)
fig = None
if len(selected_features) == 1 and poly_degree == 1:
    actual = np.squeeze(y)
    pred = np.squeeze(model.predict(X_poly))
    min_len = min(len(actual), len(pred))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X[:min_len], actual[:min_len], alpha=0.2, label='Actual')
    ax.plot(X[:min_len], pred[:min_len], color='red', linewidth=2, label=f'{model_type} Prediction')
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel('Median House Value')
    ax.set_title(f'Prediction vs. Actual: {selected_features[0]}')
    ax.legend()

# Residual plot
fig2 = None
actual_train = np.squeeze(y_train)
pred_train = np.squeeze(y_pred_train)
min_len_train = min(len(actual_train), len(pred_train))
residuals = actual_train[:min_len_train] - pred_train[:min_len_train]
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.scatter(pred_train[:min_len_train], residuals, alpha=0.2)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('Predicted Value (Train)')
ax2.set_ylabel('Residual')
ax2.set_title('Residual Plot (Train Model)')

# Loss curve (for Linear Regression)
fig3 = None
if model_type == "Linear":
    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)
    theta = np.zeros((X_train_arr.shape[1], 1))
    loss_history = []
    n_iterations = 1000 # Fixed iterations for scratch model
    for iteration in range(n_iterations):
        gradients = 2 / X_train_arr.shape[0] * X_train_arr.T.dot(X_train_arr.dot(theta) - y_train_arr)
        theta -= 0.01 * gradients # Fixed learning rate
        if iteration % 10 == 0:
            mse = np.mean((y_train_arr - X_train_arr.dot(theta)) ** 2)
            loss_history.append(mse)
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(np.arange(0, n_iterations, 10), loss_history, color='purple')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('MSE Loss')
    ax3.set_title('Loss Curve (Scratch Model)')

# Error analysis by region and price range
region_error, fig_reg, price_error, fig_price = None, None, None, None
actual = np.squeeze(y_test)
predicted = np.squeeze(y_pred_test)
min_len = min(len(actual), len(predicted))
err_df = pd.DataFrame({
    'Actual': actual[:min_len],
    'Predicted': predicted[:min_len]
})
err_df['Error'] = err_df['Predicted'] - err_df['Actual']
# Add Region column to err_df from test set
_, test_indices = train_test_split(np.arange(len(data)), test_size=test_size, random_state=42)
err_df['Region'] = data.iloc[test_indices]['Region'].values[:min_len]
if 'Region' in err_df.columns:
    region_error = err_df.groupby('Region')['Error'].mean().reset_index()
    fig_reg, ax_reg = plt.subplots()
    ax_reg.bar(region_error['Region'], region_error['Error'])
    ax_reg.set_ylabel('Mean Error')
    ax_reg.set_title('Mean Prediction Error by Region')
if 'Actual' in err_df.columns:
    err_df['PriceRange'] = pd.qcut(err_df['Actual'], 3, labels=['Low', 'Medium', 'High'])
    price_error = err_df.groupby('PriceRange')['Error'].mean().reset_index()
    fig_price, ax_price = plt.subplots()
    ax_price.bar(price_error['PriceRange'], price_error['Error'])
    ax_price.set_ylabel('Mean Error')
    ax_price.set_title('Mean Prediction Error by Price Range')

# --- Advanced Dashboard UI ---
# Load custom CSS from external file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <div class='main-header'>
        <h1>🏠 Advanced California Housing Dashboard</h1>
        <p>
            Explore, visualize, and analyze California housing data with interactive regression models and beautiful plots.
        </p>
    </div>
""", unsafe_allow_html=True)

# Tabs for dashboard sections
main_tabs = st.tabs([
    "Model & Metrics",
    "Visualizations",
    "Feature Engineering",
    "Error Analysis",
    "Custom Prediction",
    "Live Model Comparison"
])

with main_tabs[0]:
    st.subheader("Model & Metrics")
    st.write(f"**Train MSE:** {mse_train:.2f} | **MAE:** {mae_train:.2f} | **RMSE:** {rmse_train:.2f} | **R²:** {r2_train:.4f}")
    st.write(f"**Test MSE:** {mse_test:.2f} | **MAE:** {mae_test:.2f} | **RMSE:** {rmse_test:.2f} | **R²:** {r2_test:.4f}")
    st.write(f"**{cv_folds}-Fold CV R² (mean ± std):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    st.markdown("---")
    st.subheader("Downloadable Results")
    # Predictions (test set)
    actual = np.squeeze(y_test)
    predicted = np.squeeze(y_pred_test)
    min_len = min(len(actual), len(predicted))
    pred_df = pd.DataFrame({
        'Actual': actual[:min_len],
        'Predicted': predicted[:min_len]
    })
    pred_csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions (Test Set)",
        data=pred_csv,
        file_name='predictions_test.csv',
        mime='text/csv'
    )
    # Model coefficients
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': feature_names if len(feature_names) == len(model.coef_.flatten()) else [f"f{i}" for i in range(len(model.coef_.flatten()))],
            'Coefficient': model.coef_.flatten()
        })
        coef_csv = coef_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Model Coefficients",
            data=coef_csv,
            file_name='model_coefficients.csv',
            mime='text/csv'
        )
    st.markdown("---")
    st.subheader("Loss Curve (Scratch Model)")
    if model_type == "Linear":
        st.pyplot(fig3)
    st.markdown("---")
    st.subheader("Residual Plot (Train Model)")
    st.pyplot(fig2)
    st.markdown("---")
    st.subheader("Prediction vs. Actual (Single Feature)")
    if len(selected_features) == 1 and poly_degree == 1:
        st.pyplot(fig)

with main_tabs[1]:
    st.subheader("Modern Visualizations")
    st.sidebar.header("Modern Visualizations")
    plot_type = st.sidebar.selectbox("Choose plot type:", [
        "Scatter Plot (Plotly)", "Histogram (Plotly)", "KDE Distribution (Seaborn)", "Pairplot (Seaborn)", "Boxplot (Seaborn)", "Violin Plot (Seaborn)", "Swarm Plot (Seaborn)", "Feature Importance (Gradient)", "Error Distribution (Plotly)", "Correlation Heatmap (Seaborn)", "Line Plot (Plotly)", "Bar Plot (Plotly)", "Density Contour (Plotly)"
    ], key="plot_type")
    st.subheader(f"Modern Visualization: {plot_type}")

    if plot_type == "Scatter Plot (Plotly)":
        fig = px.scatter(
            data,
            x=selected_features[0] if selected_features else all_features[0],
            y="median_house_value",
            color="median_house_value",
            color_continuous_scale="Viridis",
            title="Scatter Plot: Feature vs. Median House Value",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Histogram (Plotly)":
        fig = px.histogram(
            data,
            x="median_house_value",
            nbins=40,
            color_discrete_sequence=["#636EFA"],
            title="Histogram: Median House Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "KDE Distribution (Seaborn)":
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x=data["median_house_value"], fill=True, color="#636EFA", ax=ax)
        ax.set_title("KDE Distribution: Median House Value")
        st.pyplot(fig)

    elif plot_type == "Pairplot (Seaborn)":
        import seaborn as sns
        pairplot_features = selected_features[:3] + ["median_house_value"] if len(selected_features) >= 3 else all_features[:3] + ["median_house_value"]
        # Only allow pairplot if at least two features
        if len(pairplot_features) < 2:
            st.warning("Select at least two features for pairplot.")
        else:
            pairplot_df = data[pairplot_features]
            if isinstance(pairplot_df, pd.Series):
                pairplot_df = pairplot_df.to_frame()
            if not isinstance(pairplot_df, pd.DataFrame):
                pairplot_df = pd.DataFrame(pairplot_df)
            pairplot = sns.pairplot(pairplot_df, diag_kind="kde", corner=True)
            st.pyplot(pairplot.fig)

    elif plot_type == "Boxplot (Seaborn)":
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=data["median_house_value"], ax=ax, color="#00BFFF", orient='h')
        ax.set_title("Boxplot: Median House Value")
        st.pyplot(fig)

    elif plot_type == "Violin Plot (Seaborn)":
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(x=data["median_house_value"], ax=ax, color="#FF69B4", orient='h')
        ax.set_title("Violin Plot: Median House Value")
        st.pyplot(fig)

    elif plot_type == "Swarm Plot (Seaborn)":
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.swarmplot(x=data["median_house_value"], ax=ax, color="#32CD32", orient='h')
        ax.set_title("Swarm Plot: Median House Value")
        st.pyplot(fig)

    elif plot_type == "Feature Importance (Gradient)":
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_.flatten())
            names = feature_names if len(feature_names) == len(importances) else [f"f{i}" for i in range(len(importances))]
            cmap = plt.get_cmap("viridis")
            colors = cmap(importances / (importances.max() + 1e-8))
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(names, importances, color=colors)
            ax.set_ylabel('Absolute Coefficient Value')
            ax.set_xlabel('Feature')
            ax.set_title('Feature Importance (Gradient Colors)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.warning("Feature importance is only available for linear models.")

    elif plot_type == "Error Distribution (Plotly)":
        err = np.squeeze(y_pred_test) - np.squeeze(y_test)
        fig = px.histogram(
            x=err,
            nbins=40,
            color_discrete_sequence=["#EF553B"],
            title="Error Distribution: Predicted - Actual",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap (Seaborn)":
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = data[selected_features + ['median_house_value']].select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='mako', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif plot_type == "Line Plot (Plotly)":
        fig = px.line(
            data,
            x=selected_features[0] if selected_features else all_features[0],
            y="median_house_value",
            color_discrete_sequence=["#636EFA"],
            title="Line Plot: Feature vs. Median House Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bar Plot (Plotly)":
        fig = px.bar(
            data,
            x=selected_features[0] if selected_features else all_features[0],
            y="median_house_value",
            color="median_house_value",
            color_continuous_scale="Blues",
            title="Bar Plot: Feature vs. Median House Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Density Contour (Plotly)":
        fig = px.density_contour(
            data,
            x=selected_features[0] if selected_features else all_features[0],
            y="median_house_value",
            color="median_house_value",
            title="Density Contour: Feature vs. Median House Value",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

with main_tabs[2]:
    st.subheader("Feature Engineering")
    st.sidebar.header("Feature Engineering")
    # Region clustering (coastal/inland)
    add_region = st.sidebar.checkbox("Add region clustering (coastal/inland)")
    if add_region and 'Region' not in data.columns:
        def assign_region(row):
            return 'Coastal' if row['longitude'] < -120 else 'Inland'
        data['Region'] = data.apply(assign_region, axis=1)
    if 'Region' in data.columns:
        st.sidebar.write("✅ Region clustering (coastal/inland) is already applied to the data.")

    # Interaction terms
    enable_interactions = st.sidebar.checkbox("Add interaction terms between features")
    if enable_interactions:
        from itertools import combinations
        for f1, f2 in combinations(selected_features, 2):
            col_name = f"{f1}_x_{f2}"
            data[col_name] = data[f1] * data[f2]
        # Add new interaction columns to selected_features
        selected_features += [f"{f1}_x_{f2}" for f1, f2 in combinations(selected_features, 2)]

    # Outlier detection/removal
    enable_outlier_removal = st.sidebar.checkbox("Remove outliers (3σ from mean)")
    if enable_outlier_removal:
        # Remove rows where any selected feature or target is >3 std from mean
        mask = np.ones(len(data), dtype=bool)
        for col in selected_features + ['median_house_value']:
            col_z = (data[col] - data[col].mean()) / data[col].std()
            mask &= (np.abs(col_z) < 3)
        data = data[mask]
        st.sidebar.write(f"Removed {np.sum(~mask)} outlier rows.")

    # Outlier visualization
    if st.checkbox("Show outlier visualization"):
        import seaborn as sns
        for col in selected_features:
            fig_out, ax_out = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=data[col], ax=ax_out, orient='h')
            ax_out.set_title(f'Boxplot of {col}')
            st.pyplot(fig_out)

    # Geospatial plot
    st.sidebar.header("Visualization Enhancements")
    show_geo = st.sidebar.checkbox("Show geospatial plot (map)")
    if show_geo:
        _, test_indices = train_test_split(np.arange(len(data)), test_size=test_size, random_state=42)
        plot_df = data.iloc[test_indices].copy()
        plot_df['Predicted'] = np.squeeze(y_pred_test)[:len(plot_df)]
        plot_df['Error'] = (np.squeeze(y_pred_test)[:len(plot_df)] - np.squeeze(y_test)[:len(plot_df)])
        color_col = None
        if add_region:
            region_cols = [c for c in plot_df.columns if c.startswith('region_')]
            if region_cols:
                plot_df['Region'] = np.where(plot_df[region_cols[0]] == 1, 'Coastal', 'Inland')
                color_col = 'Region'
        fig_map = px.scatter_map(
            plot_df,
            lat="latitude",
            lon="longitude",
            color=color_col if color_col else 'Error',
            size_max=8,
            zoom=5,
            map_style="carto-positron",
            hover_data={"Predicted": True, "Error": True, "median_house_value": True}
        )
        fig_map.update_layout(title="Geospatial Plot of Predictions and Errors", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # Partial Dependence Plots (PDP)
    show_pdp = st.sidebar.checkbox("Show partial dependence plots (PDP)")
    if show_pdp:
        st.subheader("Partial Dependence Plots (PDP)")
        # Only use features present in feature_names
        features_for_pdp = [f for f in selected_features[:3] if f in feature_names]  # Limit to 3 for clarity
        if not features_for_pdp:
            st.warning("No valid features available for PDP plot.")
        else:
            fig_pdp, ax_pdp = plt.subplots(1, len(features_for_pdp), figsize=(6 * len(features_for_pdp), 4))
            if len(features_for_pdp) == 1:
                ax_pdp = [ax_pdp]
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_test, features_for_pdp, feature_names=feature_names, ax=ax_pdp
                )
                st.pyplot(fig_pdp)
            except Exception as e:
                st.warning(f"Could not plot PDP: {e}")

    # Feature Importance Visualization
    show_importance = st.sidebar.checkbox("Show feature importance (coefficients)")
    if show_importance:
        st.subheader("Feature Importance (Model Coefficients)")
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_.flatten())
            # For polynomial, feature_names may be longer
            names = feature_names if len(feature_names) == len(importances) else [f"f{i}" for i in range(len(importances))]
            fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            ax_imp.bar(names, importances)
            ax_imp.set_ylabel('Absolute Coefficient Value')
            ax_imp.set_xlabel('Feature')
            ax_imp.set_title('Feature Importance (Linear Model)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_imp)
        else:
            st.warning("Feature importance is only available for linear models.")

    # SHAP explanations
    show_shap = st.sidebar.checkbox("Show SHAP explanations (feature impact)")
    if show_shap:
        st.subheader("SHAP Feature Impact (Summary Plot)")
        try:
            # Ensure feature_names matches X_test shape
            shap_feature_names = feature_names
            # Always use np.asarray to guarantee .shape exists
            import numpy as np
            X_test_arr = np.asarray(X_test)
            n_features = X_test_arr.shape[1]
            if len(feature_names) != n_features:
                shap_feature_names = [f"f{i}" for i in range(n_features)]
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            fig_shap = plt.figure()
            shap.summary_plot(shap_values, X_test, feature_names=shap_feature_names, show=False)
            st.pyplot(fig_shap, bbox_inches='tight', dpi=150)
        except Exception as e:
            st.warning(f"SHAP explanations not available for this model: {e}")

with main_tabs[3]:
    st.subheader("Error Analysis")
    st.write("**Average Prediction Error by Region:**")
    st.dataframe(region_error)
    if fig_reg is not None and region_error is not None and not region_error.empty:
        st.pyplot(fig_reg)
    else:
        st.warning("No data available for region error plot.")
    st.write("**Average Prediction Error by Price Range:**")
    st.dataframe(price_error)
    if fig_price is not None and price_error is not None and not price_error.empty:
        st.pyplot(fig_price)
    else:
        st.warning("No data available for price range error plot.")

with main_tabs[4]:
    st.subheader("Custom Data Prediction")
    st.sidebar.header("Custom Data Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload CSV for prediction", type=["csv"])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        missing_cols = [col for col in selected_features if col not in user_data.columns]
        if missing_cols:
            st.warning(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            # Feature engineering for user data (region, interactions)
            for c in [c for c in data.columns if c.startswith('region_')]:
                if c not in user_data.columns:
                    user_data[c] = 0
            if enable_interactions:
                from itertools import combinations
                for f1, f2 in combinations(selected_features, 2):
                    col_name = f"{f1}_x_{f2}"
                    user_data[col_name] = user_data[f1] * user_data[f2]
            # Normalize
            X_user = user_data[selected_features].to_numpy()
            X_user_norm = (X_user - X_mean) / X_std
            # Polynomial
            if poly is not None and poly_degree > 1:
                X_user_poly = poly.transform(X_user_norm)
            else:
                X_user_poly = X_user_norm
            if model is not None:
                user_pred = model.predict(X_user_poly)
                user_pred_df = user_data.copy()
                user_pred_df['Predicted_median_house_value'] = user_pred.flatten()
                st.subheader("Predictions for Uploaded Data")
                st.write(user_pred_df.head())
                user_pred_csv = user_pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions (Uploaded Data)",
                    data=user_pred_csv,
                    file_name='uploaded_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error("Model is not defined. Cannot predict.")

with main_tabs[5]:
    st.subheader("Live Model Comparison")
    st.sidebar.header("Live Model Comparison")
    model_options = ["Linear", "Polynomial", "Ridge", "Lasso"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model 1**")
        model1_type = st.selectbox("Model 1 Type", model_options, key="model1")
        model1_degree = st.slider("Polynomial Degree (Model 1)", 1, 5, 1, key="deg1") if model1_type == "Polynomial" else 1
        model1_alpha = st.slider("Alpha (Model 1)", 0.01, 2.0, 1.0, 0.01, key="alpha1") if model1_type in ["Ridge", "Lasso"] else 1.0
    with col2:
        st.markdown("**Model 2**")
        model2_type = st.selectbox("Model 2 Type", model_options, key="model2")
        model2_degree = st.slider("Polynomial Degree (Model 2)", 1, 5, 1, key="deg2") if model2_type == "Polynomial" else 1
        model2_alpha = st.slider("Alpha (Model 2)", 0.01, 2.0, 1.0, 0.01, key="alpha2") if model2_type in ["Ridge", "Lasso"] else 1.0
    # Prepare features
    def get_model_and_pred(model_type, degree, alpha):
        if degree > 1:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X_norm)
        else:
            X_poly = X_norm
        X_train_, X_test_, y_train_, y_test_ = train_test_split(X_poly, y, test_size=test_size, random_state=42)
        if model_type == "Linear":
            m = LinearRegression()
        elif model_type == "Polynomial":
            m = LinearRegression()
        elif model_type == "Ridge":
            m = Ridge(alpha=alpha)
        elif model_type == "Lasso":
            m = Lasso(alpha=alpha, max_iter=10000)
        else:
            st.error("Unknown model type selected.")
            return None, None, None, None, None, None, None
        m.fit(X_train_, y_train_)
        y_pred_train_ = m.predict(X_train_)
        y_pred_test_ = m.predict(X_test_)
        return m, X_train_, X_test_, y_train_, y_test_, y_pred_train_, y_pred_test_
    m1, Xtr1, Xte1, ytr1, yte1, yptr1, ypte1 = get_model_and_pred(model1_type, model1_degree, model1_alpha)
    m2, Xtr2, Xte2, ytr2, yte2, yptr2, ypte2 = get_model_and_pred(model2_type, model2_degree, model2_alpha)
    # Metrics
    m1_metrics = get_metrics(yte1, ypte1)
    m2_metrics = get_metrics(yte2, ypte2)
    st.write("**Test Set Metrics:**")
    st.write(f"Model 1 ({model1_type}): MSE={m1_metrics[0]:.2f}, MAE={m1_metrics[1]:.2f}, RMSE={m1_metrics[2]:.2f}, R²={m1_metrics[3]:.4f}")
    st.write(f"Model 2 ({model2_type}): MSE={m2_metrics[0]:.2f}, MAE={m2_metrics[1]:.2f}, RMSE={m2_metrics[2]:.2f}, R²={m2_metrics[3]:.4f}")
    # Plots (for single feature)
    if len(selected_features) == 1 and model1_degree == 1 and model2_degree == 1:
        fig_cmp, ax_cmp = plt.subplots(figsize=(8, 5))
        ax_cmp.scatter(X, y, alpha=0.2, label='Actual')
        if m1 is not None:
            ax_cmp.plot(X, m1.predict(X_norm), color='red', linewidth=2, label=f'Model 1 ({model1_type})')
        if m2 is not None:
            ax_cmp.plot(X, m2.predict(X_norm), color='green', linewidth=2, linestyle='--', label=f'Model 2 ({model2_type})')
        ax_cmp.set_xlabel(selected_features[0])
        ax_cmp.set_ylabel('Median House Value')
        ax_cmp.set_title('Model Comparison')
        ax_cmp.legend()
        st.pyplot(fig_cmp) 

# Custom footer
st.markdown("""
    <div style='background: linear-gradient(90deg, #FFD369 0%, #00BFFF 100%);padding:18px 0 18px 0;border-radius:12px;margin-top:32px;text-align:center;display:flex;justify-content:center;align-items:center;gap:32px;'>
        <span style='font-size:20px;font-weight:800;color:#222831;text-shadow:0 1px 6px rgba(255,255,255,0.25),0 1px 1px rgba(0,0,0,0.10);letter-spacing:0.5px;'>Project by <span style='color:#222831;font-weight:900;text-shadow:0 2px 8px #FFD369,0 1px 1px rgba(0,0,0,0.10);'>Himanshu Salunke</span></span>
        <a href='https://github.com/HRS0221/' target='_blank' rel='noopener noreferrer'
            style='display:inline-block;padding:10px 28px;background:#222831;color:#FFD369;font-size:17px;border:none;border-radius:6px;text-decoration:none;font-weight:600;box-shadow:0 2px 8px rgba(0,0,0,0.08);transition:background 0.2s;'>
            GitHub
        </a>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Model Persistence")
if st.sidebar.button("Save Trained Model"):
    try:
        joblib.dump(model, "trained_model.joblib")
        st.sidebar.success("Model saved as trained_model.joblib")
    except Exception as e:
        st.sidebar.error(f"Failed to save model: {e}")
if st.sidebar.button("Load Model"):
    try:
        loaded_model = joblib.load("trained_model.joblib")
        model = loaded_model
        st.sidebar.success("Model loaded from trained_model.joblib")
        # Optionally, update predictions
        y_pred_test = model.predict(X_test)
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")