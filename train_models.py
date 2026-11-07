
import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


# ======================================================
# SUPPRESS WARNINGS
# ======================================================

warnings.filterwarnings("ignore")


# ======================================================
# CREATE FOLDERS
# ======================================================

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ======================================================
# CONFIGURATION
# ======================================================

RUN_GRID_SEARCH = False

RANDOM_STATE = 42


# ======================================================
# COLUMN NAMES FOR ADULT DATASET
# ======================================================

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


# ======================================================
# LOAD DATASET
# ======================================================

print("\nLoading Dataset...\n")

df = pd.read_csv(
    "data/adult.csv",
    header=None,
    names=columns,
    skipinitialspace=True
)

print("Original Shape:", df.shape)


# ======================================================
# HANDLE MISSING VALUES
# ======================================================

df.replace("?", pd.NA, inplace=True)

df.dropna(inplace=True)

# Remove unnecessary spaces
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Normalize target values
df["income"] = (
    df["income"]
    .str.rstrip(".")
    .str.strip()
)

print("Shape After Cleaning:", df.shape)


# ======================================================
# FEATURES & TARGET
# ======================================================

X = df.drop("income", axis=1)

y = df["income"].apply(
    lambda x: 1 if x == ">50K" else 0
)


# ======================================================
# CHECK CLASS DISTRIBUTION
# ======================================================

print("\nTarget Distribution:")

print(y.value_counts())

print("\nTarget Percentage:")

print(y.value_counts(normalize=True) * 100)


# ======================================================
# IDENTIFY COLUMNS
# ======================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ======================================================
# PREPROCESSOR
# ======================================================

def create_preprocessor(dense=False):

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ],
        sparse_threshold=0.0 if dense else 0.3
    )


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ======================================================
# CROSS VALIDATION
# ======================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ======================================================
# MODELS (TUNED)
# ======================================================

models = {

    "Logistic_Regression": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            )
        )
    ]),

    "Decision_Tree": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=6,
                min_samples_split=30,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="manhattan",
                n_jobs=-1
            )
        )
    ]),

    "Naive_Bayes": Pipeline([
        (
            "preprocessor",
            create_preprocessor(dense=True)
        ),
        (
            "classifier",
            GaussianNB(
                var_smoothing=1e-8
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

}


# ======================================================
# HYPERPARAMETER GRID
# ======================================================

param_grids = {

    "Logistic_Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"]
    },

    "Decision_Tree": {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [4, 6, 8, 10],
        "classifier__min_samples_split": [20, 30, 50],
        "classifier__min_samples_leaf": [5, 10, 15]
    },

    "KNN": {
        "classifier__n_neighbors": [5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"]
    },

    "Naive_Bayes": {
        "classifier__var_smoothing": [
            1e-11, 1e-10, 1e-9, 1e-8, 1e-7
        ]
    },

    "SVM": {
        "classifier__C": [0.5, 1, 5],
        "classifier__gamma": ["scale", 0.01]
    }

}


# ======================================================
# TRAIN & EVALUATE
# ======================================================

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name} ...")

    # ----------------------------------------------
    # GRID SEARCH OR NORMAL TRAINING
    # ----------------------------------------------

    if RUN_GRID_SEARCH:

        print("Running GridSearchCV...")

        grid = GridSearchCV(
            model,
            param_grids[model_name],
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        print("Best Parameters:", grid.best_params_)

        print(
            "Best CV F1:",
            round(grid.best_score_, 4)
        )

    else:

        best_model = model

        best_model.fit(
            X_train,
            y_train
        )

    # ----------------------------------------------
    # PREDICTIONS
    # ----------------------------------------------

    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    # ----------------------------------------------
    # EVALUATION METRICS
    # ----------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # ----------------------------------------------
    # STORE RESULTS
    # ----------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": round(accuracy, 4),

        "AUC": round(auc, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1": round(f1, 4),

        "MCC": round(mcc, 4)

    })

    # ----------------------------------------------
    # SAVE MODEL PIPELINE
    # ----------------------------------------------

    joblib.dump(
        best_model,
        f"model/{model_name}.pkl"
    )

    print(f"{model_name} model saved")


# ======================================================
# RESULTS TABLE
# ======================================================

results_df = pd.DataFrame(results)

results_df.sort_values(
    by="Accuracy",
    ascending=False,
    inplace=True
)


# ======================================================
# SAVE CSV
# ======================================================

results_df.to_csv(
    "results/metrics.csv",
    index=False
)


# ======================================================
# DISPLAY RESULTS
# ======================================================

print("\n")
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(results_df.to_string(index=False))


# ======================================================
# IDENTIFY BEST MODELS
# ======================================================

best_accuracy = results_df.iloc[0]["Model"]

best_f1 = results_df.loc[
    results_df["F1"].idxmax(),
    "Model"
]

best_mcc = results_df.loc[
    results_df["MCC"].idxmax(),
    "Model"
]

best_auc = results_df.loc[
    results_df["AUC"].idxmax(),
    "Model"
]

print("\nBest Accuracy Model:", best_accuracy)

print("Best F1 Model:", best_f1)

print("Best MCC Model:", best_mcc)

print("Best AUC Model:", best_auc)


# ======================================================
# COMPLETION
# ======================================================

print("\nMetrics saved to:")
print("results/metrics.csv")

print("\nModels saved to:")
print("model/")

print("\nTraining Completed Successfully!")
