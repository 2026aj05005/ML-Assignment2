
import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Machine Learning Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Classification Dashboard")

st.markdown(
    """
    ### Adult Income Prediction

    This application predicts whether an individual's
    annual income is above or below $50K using
    five Machine Learning classification algorithms.
    """
)


# ==================================================
# DATASET COLUMN NAMES
# ==================================================

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

feature_columns = columns[:-1]


# ==================================================
# MODEL PATHS
# ==================================================

MODEL_PATHS = {

    "Logistic Regression":
        "model/Logistic_Regression.pkl",

    "Decision Tree":
        "model/Decision_Tree.pkl",

    "KNN":
        "model/KNN.pkl",

    "Naive Bayes":
        "model/Naive_Bayes.pkl",

    "SVM":
        "model/SVM.pkl"

}


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Settings")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Available Models**

    1. Logistic Regression
    2. Decision Tree
    3. KNN
    4. Naive Bayes
    5. SVM
    """
)


# ==================================================
# SHOW TRAINING RESULTS
# ==================================================

if os.path.exists("results/metrics.csv"):

    with st.expander("View Training Results"):

        training_results = pd.read_csv(
            "results/metrics.csv"
        )

        st.dataframe(
            training_results,
            use_container_width=True
        )


# ==================================================
# LOAD AND CLEAN TEST DATASET
# ==================================================

def load_test_dataset(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        skipinitialspace=True,
        comment="|"
    )

    # Normalize column names
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Check whether the CSV contains headers.
    # If not, read it again using Adult dataset columns.

    if not set(columns).issubset(set(df.columns)):

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            header=None,
            names=columns,
            skipinitialspace=True,
            comment="|"
        )

    # Normalize column names again
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Remove extra whitespace from text values
    for col in df.select_dtypes(
        include=["object"]
    ).columns:

        df[col] = df[col].str.strip()

    # Handle missing values
    df.replace(
        r"^\s*\?\s*$",
        pd.NA,
        regex=True,
        inplace=True
    )

    # Validate required feature columns
    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep the original dataset row numbers
    df["_original_row"] = df.index

    # Identify whether actual target is available
    has_target = "income" in df.columns

    # Remove rows with missing features
    required_columns = feature_columns.copy()

    if has_target:
        required_columns.append("income")

    df.dropna(
        subset=required_columns,
        inplace=True
    )

    if df.empty:

        raise ValueError(
            "No valid records found after cleaning."
        )

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df, has_target


# ==================================================
# PROCESS UPLOADED DATASET
# ==================================================

if uploaded_file is not None:

    try:

        # ------------------------------------------
        # LOAD DATASET
        # ------------------------------------------

        df, has_target = load_test_dataset(
            uploaded_file
        )

        st.subheader("Dataset Preview")

        st.write(
            "Dataset Shape:",
            df.shape
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # ------------------------------------------
        # FEATURES
        # ------------------------------------------

        X = df[feature_columns].copy()

        # ------------------------------------------
        # LOAD SELECTED MODEL
        # ------------------------------------------

        model_path = MODEL_PATHS[selected_model]

        if not os.path.exists(model_path):

            st.error(
                f"Model file not found: {model_path}"
            )

            st.stop()

        model = load_model(model_path)

        # ------------------------------------------
        # PREDICTIONS
        # ------------------------------------------

        # Pipeline handles preprocessing internally.
        # Do not apply a separate preprocessor.

        y_pred = model.predict(X)

        y_prob = model.predict_proba(X)[:, 1]

        # ------------------------------------------
        # PREDICTION RESULTS
        # ------------------------------------------

        st.subheader(
            f"Prediction Results - {selected_model}"
        )

        result_df = X.copy()

        result_df["Predicted Income"] = [
            ">50K" if value == 1 else "<=50K"
            for value in y_pred
        ]

        result_df["Probability >50K"] = y_prob

        # ------------------------------------------
        # EVALUATION METRICS
        # ------------------------------------------

        if has_target:

            st.subheader(
                f"Evaluation Metrics - {selected_model}"
            )

            # Normalize actual income labels

            y_actual_text = (
                df["income"]
                .astype(str)
                .str.strip()
                .str.rstrip(".")
            )

            valid_labels = {
                "<=50K": 0,
                ">50K": 1
            }

            unknown_labels = set(
                y_actual_text.unique()
            ) - set(valid_labels.keys())

            if unknown_labels:

                raise ValueError(
                    f"Unknown income labels: {unknown_labels}"
                )

            y = y_actual_text.map(
                valid_labels
            ).astype(int)

            # --------------------------------------
            # CALCULATE METRICS
            # --------------------------------------

            accuracy = accuracy_score(
                y,
                y_pred
            )

            precision = precision_score(
                y,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y,
                y_pred,
                zero_division=0
            )

            auc = roc_auc_score(
                y,
                y_prob
            ) if y.nunique() == 2 else None

            mcc = matthews_corrcoef(
                y,
                y_pred
            )

            # --------------------------------------
            # DISPLAY METRICS
            # --------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col3.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col1.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}" if auc is not None else "N/A"
            )

            col3.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            if auc is None:

                st.warning(
                    "AUC cannot be calculated because "
                    "the uploaded dataset contains only one actual class."
                )

            # --------------------------------------
            # CONFUSION MATRIX
            # --------------------------------------

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"],
                ax=ax
            )

            ax.set_xlabel("Predicted Income")
            ax.set_ylabel("Actual Income")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

            plt.close(fig)

            # --------------------------------------
            # CLASSIFICATION REPORT
            # --------------------------------------

            st.subheader(
                "Classification Report"
            )

            report = classification_report(
                y,
                y_pred,
                labels=[0, 1],
                target_names=["<=50K", ">50K"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

            # Add actual values to download
            result_df["Actual Income"] = [
                ">50K" if value == 1 else "<=50K"
                for value in y
            ]

        else:

            st.info(
                "No actual income column found. "
                "Displaying predictions only; evaluation metrics "
                "cannot be calculated."
            )

        # ------------------------------------------
        # PREDICTION DISTRIBUTION
        # ------------------------------------------

        st.subheader(
            "Prediction Distribution"
        )

        pred_count = pd.DataFrame({

            "Income Class": [
                "<=50K",
                ">50K"
            ],

            "Count": [
                (y_pred == 0).sum(),
                (y_pred == 1).sum()
            ]

        })

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted <=50K",
            int((y_pred == 0).sum())
        )

        col2.metric(
            "Predicted >50K",
            int((y_pred == 1).sum())
        )

        st.dataframe(
            pred_count,
            use_container_width=True
        )

        # ------------------------------------------
        # DOWNLOAD PREDICTIONS
        # ------------------------------------------

        st.subheader(
            "Download Prediction Results"
        )

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="income_predictions.csv",
            mime="text/csv"
        )

        st.success(
            f"{selected_model} predictions completed successfully!"
        )

    except Exception as e:

        st.error(
            f"Error processing dataset: {str(e)}"
        )

else:

    st.info(
        "Please upload your Adult Income test CSV "
        "using the sidebar to begin prediction."
    )

    st.subheader("Available Classification Models")

    model_info = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "SVM"
        ],

        "Algorithm Type": [
            "Linear Classification",
            "Tree-Based Classification",
            "Distance-Based Classification",
            "Probabilistic Classification",
            "Support Vector Classification"
        ]

    })

    st.dataframe(
        model_info,
        use_container_width=True
    )