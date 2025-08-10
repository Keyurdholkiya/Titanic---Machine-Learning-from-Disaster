import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MODEL_FILE = "titanic_model.pkl"

# Function to clean raw Titanic data
def clean_data(df):
    # Fill missing Embarked
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Fill missing Cabin with "Unknown"
    df["Cabin"].fillna("Unknown", inplace=True)

    # Fill missing Age & Fare
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Extract Title from Name
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
    df["Title"] = df["Title"].replace(['Mme'], 'Mrs')

    # Drop unused columns
    df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True, errors="ignore")

    return df

if not os.path.exists(MODEL_FILE):
    # ================== TRAIN MODEL ==================
    print("Training model...")
    df = pd.read_csv("train.csv")
    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    X = clean_data(X)

    # Features by type
    categorical_features = ["Sex", "Embarked", "Title"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Preprocessing + Model
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Train/Test split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_val, y_pred))

    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

else:
    # ================== LOAD & PREDICT ==================
    print("Loading model...")
    model = joblib.load(MODEL_FILE)

    # Load and clean test.csv
    df_test = pd.read_csv("test.csv")
    df_test_clean = clean_data(df_test)

    # Predict
    predictions = model.predict(df_test_clean)
    print("Predictions for test.csv:")
    print(predictions)
    df_test_clean["Survived_prediction"] = predictions
    # df_test_clean["Survived"] = 
    df_test_clean.to_csv("testing.csv",index=False)
