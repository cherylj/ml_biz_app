INPUT_FEATURES = ['Zip Code', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
       'Phone Service', 'Paperless Billing', 'Monthly Charges',
       'CLTV', 'Gender', 'Contract', 'Payment Method',
       'Multiple Lines', 'Online Security', 'Online Backup', 
       'Device Protection', 'Tech Support', 'Internet Service',
       'Streaming TV', 'Streaming Movies']

MODEL_FEATURES = ['Zip Code', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
       'Phone Service', 'Paperless Billing', 'Monthly Charges',
       'CLTV', 'Gender_Female', 'Gender_Male', 'Contract_Month-to-month',
       'Contract_Two year', 'Payment Method_Bank transfer (automatic)',
       'Payment Method_Credit card (automatic)',
       'Payment Method_Electronic check', 'Payment Method_Mailed check',
       'Multiple Lines_No', 'Multiple Lines_Yes', 'Online Security_No',
       'Online Security_Yes', 'Online Backup_No', 'Online Backup_Yes',
       'Device Protection_Yes', 'Tech Support_No', 'Tech Support_Yes',
       'Internet Service_DSL', 'Internet Service_Fiber optic',
       'Internet Service_No', 'Streaming TV_No', 'Streaming TV_Yes',
       'Streaming Movies_No', 'Streaming Movies_Yes']

FEATURES_TO_DROP = ['Multiple Lines_No phone service', 'Online Security_No internet service',
             'Online Backup_No internet service', 'Device Protection_No internet service',
             'Tech Support_No internet service', 'Streaming TV_No internet service',
             'Streaming Movies_No internet service', 'Device Protection_No', 'Contract_One year']

YES_NO_FEATURES = ['Senior Citizen', 'Dependents', 'Partner', 'Paperless Billing',
                   'Phone Service']
MULTI_CAT_FEATURES = ['Gender', 'Contract', 'Payment Method', 'Multiple Lines',
                      'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                      'Internet Service', 'Streaming TV', 'Streaming Movies']

NUMERIC_FEATURES = ['Tenure Months', 'Monthly Charges', 'CLTV']