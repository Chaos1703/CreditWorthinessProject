import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# 1. Load your original dataset
df = pd.read_csv(
    r'C:\Users\KrishnaWali\Downloads\statlog+german+credit+data\german - Copy.data',
    header=None,
    sep=r'\s+'
)

# 2. Assign column names
df.columns = [
    'status_checking_account', 'duration_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors_guarantors', 'present_residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'number_existing_credits', 'job',
    'people_liable', 'telephone', 'foreign_worker', 'class'
]

# 3. Define metadata and override column types
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

override_sdtypes = {
    'status_checking_account': 'categorical',
    'credit_history':           'categorical',
    'purpose':                  'categorical',
    'savings_account_bonds':    'categorical',
    'present_employment_since': 'categorical',
    'personal_status_sex':      'categorical',
    'other_debtors_guarantors': 'categorical',
    'property':                 'categorical',
    'other_installment_plans':  'categorical',
    'housing':                  'categorical',
    'job':                      'categorical',
    'telephone':                'categorical',
    'foreign_worker':           'categorical',
    'class':                    'categorical'
}
for col, sdtype in override_sdtypes.items():
    metadata.update_column(column_name=col, sdtype=sdtype)

# 4. Train synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df)

# 5. Sample synthetic data (to cover both classes)
synthetic_df = synthesizer.sample(2000)

# 6. Create a balanced dataset of 500 rows (250 good + 250 bad)
good = synthetic_df[synthetic_df['class'] == 1].sample(n=250, replace=True)
bad  = synthetic_df[synthetic_df['class'] == 2].sample(n=250, replace=True)
balanced_500 = pd.concat([good, bad]).sample(frac=1).reset_index(drop=True)

# 7. Map target and drop original class
balanced_500['target'] = balanced_500['class'].map({1: 1, 2: 0})
balanced_500 = balanced_500.drop(columns=['class'])

# 8. Export
balanced_500.to_csv('german_credit_synthetic_test_500.csv', index=False)

print("✅ Done! Exported test dataset with shape:", balanced_500.shape)
print(balanced_500['target'].value_counts())
