from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os

# Updated clean_and_export including numeric imputation
DATA_FILE = r"C:\Users\KrishnaWali\Downloads\german_credit_synthetic_balanced.csv"
DATA_COLUMNS = [
    'status_checking_account', 'duration_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors_guarantors', 'present_residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'number_existing_credits', 'job',
    'people_liable', 'telephone', 'foreign_worker', 'target'
]

def clean_and_export(data_path = ""):
    file_to_read = data_path if data_path else DATA_FILE
    # 1. Read the data
    df = pd.read_csv(file_to_read, sep=",", header=0, names=DATA_COLUMNS, dtype=str, 
                     na_values=["?", "NA", "nan", "NULL" , ""], encoding="ISO-8859-1")

    # 3. Normalize and strip categorical text
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
    
    # 4. Fill object-type NaNs with "missing"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")
    
    # 6. Adding Changes to the dataset for specific models
    df = df.drop_duplicates()
    df['log_credit_amount'] = np.log1p(df['credit_amount'])
    df['log_duration_month'] = np.log1p(df['duration_month'])
    df['monthly_payment'] = pd.to_numeric(df['credit_amount'], errors='coerce') / \
                            pd.to_numeric(df['duration_month'], errors='coerce').replace(0, np.nan)


    # 7. Identify numeric-like columns by attempting conversion
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
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


    df["target"] = df["target"].astype(int)
    
    # 9. Export cleaned file
    raw_folder, raw_name = os.path.split(DATA_FILE)
    base_name, _ext = os.path.splitext(raw_name)
    cleaned_filename = f"{base_name}_cleaned_optimized.csv"
    cleaned_path = os.path.join(raw_folder, cleaned_filename)
    df.to_csv(cleaned_path, index=False)

    return cleaned_path
