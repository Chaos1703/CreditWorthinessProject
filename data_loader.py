# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import clean_and_export, DATA_COLUMNS
from sklearn.impute import SimpleImputer

def collapse_rare_categories(df, col, threshold=0.05):
    counts = df[col].value_counts(normalize=True)
    rare = counts[counts < threshold].index
    df[col] = df[col].replace(rare, "Other")
    return df

def load_and_prepare_data(model_type , return_split = True , data_path = ""):
    cleaned_path = clean_and_export(data_path)
    df = pd.read_csv(cleaned_path, low_memory=False)
    X = df.drop('target', axis=1)
    Y = df['target']
    df = collapse_rare_categories(df, "other_debtors_guarantors")
    df = collapse_rare_categories(df, "foreign_worker")
    if model_type == 'lightgbm':
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')

    elif model_type == 'randomforest':
        X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns, drop_first=True)

    elif model_type == 'catboost':
        object_cols = X.select_dtypes(include='object').columns.tolist()

        for col in object_cols:
            X[col] = X[col].astype(str)

        cat_features = [X.columns.get_loc(col) for col in object_cols]
        if(return_split):
            return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y), cat_features
        else:
            return X,Y,cat_features

    elif model_type == 'tabnet':

        cat_idxs = []
        cat_dims = []
        
        categorical_columns = X.select_dtypes(include='object').columns.tolist()

        for col in categorical_columns:
            le = LabelEncoder()
            X.loc[:, col] = le.fit_transform(X[col])
            cat_idxs.append(X.columns.get_loc(col))
            cat_dims.append(len(le.classes_))

        X_train, X_test, Y_train, Y_test = train_test_split(
            X.values, Y.values, test_size=0.2, random_state=42, stratify=Y
        )

        imputer = SimpleImputer(strategy='median')

        imputer.fit(X_train)

        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        if(return_split):
            return (X_train_imputed, X_test_imputed, Y_train, Y_test), cat_idxs, cat_dims
        else:
            X_imputed = imputer.fit_transform(X.values)
            return X_imputed, Y.values, cat_idxs, cat_dims
        
    if(return_split):
        return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    else:
        return X,Y
