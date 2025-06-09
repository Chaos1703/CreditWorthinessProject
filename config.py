from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Define the imputation function as before

def impute_numerical_knn(df, numeric_cols=None, n_neighbors=5):
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure numeric conversion
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract only numeric data
    num_data = df[numeric_cols]

    # Step 1: Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_data)

    # Step 2: KNN Impute on standardized data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled = imputer.fit_transform(scaled_data)

    # Step 3: Inverse transform to original scale
    imputed_unscaled = scaler.inverse_transform(imputed_scaled)

    # Update DataFrame
    df[numeric_cols] = imputed_unscaled

    return df

# Updated clean_and_export including numeric imputation
DATA_FILE = r"C:\Users\KrishnaWali\Downloads\german_credit_synthetic_balanced.csv"
DATA_COLUMNS = [
    'status_checking_account', 'duration_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors_guarantors', 'present_residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'number_existing_credits', 'job',
    'people_liable', 'telephone', 'foreign_worker', 'target'
]

def clean_and_export():
    # 1. Read the data
    df = pd.read_csv(DATA_FILE, sep=",", header=0, names=DATA_COLUMNS, dtype=str, 
                     na_values=["?", "NA", "nan", "NULL" , ""], encoding="ISO-8859-1")

    # 3. Normalize and strip categorical text
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
    
    # 4. Fill object-type NaNs with "missing"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")
    
    # 5. Identify numeric-like columns by attempting conversion
    numeric_cols = []
    for col in df.columns:
        if col == "target":
            continue
        # Try converting to numeric
        converted = pd.to_numeric(df[col], errors='coerce')
        # If at least 90% of non-null values are numeric, treat as numeric
        non_null = df[col].notna().sum()
        numeric_count = converted.notna().sum()
        if non_null > 0 and (numeric_count / non_null) >= 0.9:
            numeric_cols.append(col)
    
    # 6. Impute numeric NaNs using the random median ± x method
    df = impute_numerical_knn(df, numeric_cols=numeric_cols)

    df["target"] = df["target"].astype(int)
    
    # 9. Export cleaned file
    raw_folder, raw_name = os.path.split(DATA_FILE)
    base_name, _ext = os.path.splitext(raw_name)
    cleaned_filename = f"{base_name}_cleaned.csv"
    cleaned_path = os.path.join(raw_folder, cleaned_filename)
    df.to_csv(cleaned_path, index=False)

    return cleaned_path
