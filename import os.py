import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Title & Sidebar navigation
# -----------------------------
st.title("France Electricity Grid — Data Project")
st.sidebar.title("Table of contents")
pages = ["Exploration", "DataVizualization", "Modelling", "Report (PDF)"]
page = st.sidebar.radio("Go to", pages)

# -----------------------------
# Data loading (with cache)
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None, path=None, sep=","):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, sep=sep)
        if path and os.path.exists(path):
            return pd.read_csv(path, sep=sep)
    except Exception as e:
        st.warning(f"Loading error: {e}")
    return pd.DataFrame()

st.sidebar.subheader("Data source")
uploaded_csv = st.sidebar.file_uploader("Upload your CSV (recommended)", type=["csv"])
csv_sep = st.sidebar.selectbox("CSV separator", [",", ";", "\t"], index=0)
custom_path = st.sidebar.text_input("...or enter a local CSV path", value="")

df = load_data(uploaded_csv, custom_path, sep=csv_sep)

if df.empty:
    st.info("No data loaded yet. Upload your CSV from the sidebar or set a valid path, then use the pages.")
else:
    st.sidebar.write("**Columns detected:**")
    st.sidebar.write(list(df.columns))

# -----------------------------
# Page 1 — Exploration
# -----------------------------
if page == pages[0]:
    st.write("### Presentation of data")
    if not df.empty:
        st.dataframe(df.head(10))
        st.write("**Shape (rows, columns):**")
        st.write(df.shape)

        if st.checkbox("Show numeric summary (describe)"):
            st.dataframe(df.describe())

        if st.checkbox("Show missing values (per column)"):
            st.dataframe(df.isna().sum())

        preview_col = st.selectbox("Preview a column (first 25 values)", options=df.columns)
        st.write(pd.DataFrame({preview_col: df[preview_col].head(25)}))
    else:
        st.write("Please load a CSV first from the left sidebar.")

# -----------------------------
# Page 2 — DataVizualization
# -----------------------------
if page == pages[1]:
    st.write("### DataVizualization")

    if not df.empty:
        # Categorical countplot
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
        if len(cat_cols) > 0:
            chosen_cat = st.selectbox("Countplot — choose a categorical column", options=cat_cols)
            fig = plt.figure()
            sns.countplot(x=chosen_cat, data=df)
            plt.title(f"Distribution of {chosen_cat}")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.info("No categorical columns detected for a countplot.")

        # Numeric histogram
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) > 0:
            chosen_num = st.selectbox("Histogram — choose a numeric column", options=num_cols)
            bins = st.slider("Number of bins", 5, 100, 30)
            fig2 = plt.figure()
            sns.histplot(df[chosen_num].dropna(), bins=bins)
            plt.title(f"Distribution of {chosen_num}")
            st.pyplot(fig2)
        else:
            st.info("No numeric columns detected for a histogram.")

    else:
        st.write("Please load a CSV first from the left sidebar.")

# -----------------------------
# Page 3 — Modelling
# -----------------------------
if page == pages[2]:
    st.write("### Modelling")
    if not df.empty:
        target = st.selectbox("Choose target column", options=df.columns)

        is_regression = np.issubdtype(df[target].dropna().dtype, np.number)
        default_features = [c for c in df.columns if c != target]
        features = st.multiselect("Choose feature columns", options=[c for c in df.columns if c != target],
                                  default=default_features)

        if len(features) == 0:
            st.info("Please choose at least one feature.")
        else:
            work_df = df[features + [target]].copy()
            work_df = work_df.dropna(subset=[target])

            num_cols = [c for c in features if np.issubdtype(work_df[c].dtype, np.number)]
            cat_cols = [c for c in features if c not in num_cols]

            for c in num_cols:
                work_df[c] = work_df[c].fillna(work_df[c].median())
            if cat_cols:
                work_df = pd.get_dummies(work_df, columns=cat_cols, dummy_na=True)

            X = work_df.drop(columns=[target])
            y = work_df[target]

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

            if is_regression:
                choice = ['Linear Regression', 'Ridge', 'Random Forest Regressor']
                option = st.selectbox('Choice of the model', choice)

                from sklearn.linear_model import LinearRegression, Ridge
                from sklearn.ensemble import RandomForestRegressor
                if option == 'Linear Regression':
                    clf = LinearRegression()
                elif option == 'Ridge':
                    clf = Ridge()
                else:
                    clf = RandomForestRegressor(random_state=123)

                clf.fit(X_train, y_train)

                metric = st.radio('What do you want to show ?', ('R2', 'MAE'))
                from sklearn.metrics import r2_score, mean_absolute_error
                y_pred = clf.predict(X_test)
                if metric == 'R2':
                    st.write(r2_score(y_test, y_pred))
                else:
                    st.write(mean_absolute_error(y_test, y_pred))

            else:
                choice = ['Logistic Regression', 'SVC', 'Random Forest Classifier']
                option = st.selectbox('Choice of the model', choice)

                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.ensemble import RandomForestClassifier

                if option == 'Logistic Regression':
                    clf = LogisticRegression(max_iter=200)
                elif option == 'SVC':
                    clf = SVC()
                else:
                    clf = RandomForestClassifier(random_state=123)

                if y.dtype == "O":
                    y = y.astype("category").cat.codes
                    y_train = y_train.astype("category").cat.codes
                    y_test = y_test.astype("category").cat.codes

                clf.fit(X_train, y_train)

                metric = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
                from sklearn.metrics import confusion_matrix, accuracy_score
                y_pred = clf.predict(X_test)
                if metric == 'Accuracy':
                    st.write(accuracy_score(y_test, y_pred))
                else:
                    st.dataframe(confusion_matrix(y_test, y_pred))
    else:
        st.write("Please load a CSV first from the left sidebar.")

# -----------------------------
# Page 4 — Report (PDF)
# -----------------------------
if page == pages[3]:
    st.write("### Report (PDF)")
    st.write("Upload your final report (.pdf) to view or download it below.")

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf_file is not None:
        pdf_bytes = pdf_file.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf")
    else:
        st.info("No PDF uploaded yet.")
