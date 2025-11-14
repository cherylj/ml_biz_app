'''
This file will read in the Telco_customer_churn.xlsx file
in the data directory and perform the following operations:
    1. Remove irrelevant features.
    2. Update Yes/No columns to use 0/1.
    3. Create one-hot encodings for categorical values.
    4. Save the cleaned dataset in data/cleaned_telco_data.csv.
'''

import pandas as pd
from model_features import YES_NO_FEATURES, MULTI_CAT_FEATURES

data_dir = "data"
ibm_churn_df = pd.read_excel(data_dir + "/Telco_customer_churn.xlsx")

# We're going to drop the columns that aren't relevant to the classification
# or are redundant
raw_features_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Lat Long',
                    'Latitude', 'Longitude', 'Total Charges', 'Churn Label',
                    'Churn Score', 'Churn Reason']

churn_df = ibm_churn_df.drop(raw_features_to_drop, axis=1)

# Rename Churn Value for ease of use
churn_df = churn_df.rename(columns={'Churn Value': 'Churn'})

for col in YES_NO_FEATURES:
  # Change the "Yes" to a 1, and "No" to a 0
  churn_df[col] = churn_df[col].map({'Yes': 1, 'No': 0})

# Not all models will require one hot encoding, so let's save a copy of
# the cleaned data without OHE as well
churn_df.to_csv(data_dir + '/cleaned_telco_data_no_OHE.csv')

# Next, use one-hot encoding for the categorical features
churn_df = pd.get_dummies(churn_df, columns=MULTI_CAT_FEATURES, dtype=int)

# Save the cleaned dataset to a new csv file
churn_df.to_csv(data_dir + '/cleaned_telco_data.csv')