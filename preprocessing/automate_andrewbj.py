import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN

def remove_outliers(df, numerical_cols):
    """
    Removes rows containing outliers from a DataFrame based on the IQR method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - numerical_cols (List[str]): List of numerical columns to check for outliers.

    Returns:
    - pd.DataFrame: DataFrame without rows containing outliers in the specified columns.
    """

    # Calculate the Q1, Q3, and IQR value
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Filter to exclude the rows that contains outliers
    outliers_filter = ~((df[numerical_cols] < (Q1 - 1.5 * IQR)) |
                        (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    # Apply the outliers filter
    return df[outliers_filter]

def preprocess_data(input_path, output_path):
    """
    Preprocesses the patient dataset by cleaning, encoding, removing outliers,
    and handling class imbalance using SMOTEENN. Saves the processed data to a CSV file.

    Parameters:
    - input_path (str): Path to the input CSV file.
    - output_path (str): Path to save the cleaned output CSV file.

    Returns:
    - X_train (np.ndarray): Normalized training features.
    - X_test (np.ndarray): Normalized test features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    """

    # Load the data
    df = pd.read_csv(input_path)
    print(f'Data successfully loaded from: {input_path}')

    # Drop duplicates data
    df = df.drop_duplicates()
    if df.empty:
        raise ValueError('Dataset is empty after removing duplicates.')
    else:
        print(f'Duplicate data after cleaning: {df.duplicated().sum()}')

    # Merge all of the race features
    race_cols = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian',
                'race:Hispanic', 'race:Other']
    race = df[race_cols].idxmax(axis=1).str.replace('race:', '')
    df.insert(2, 'race', race)
    df.drop(columns=race_cols, inplace=True)

    # Drop the clinical notes features that will not be used in this project
    df.drop('clinical_notes', inplace=True, axis=1)

    # Split the diabetes and non diabetes patient
    df_diabet = df[df['diabetes'] == 1]
    df_ndiabet = df[df['diabetes'] == 0]

    # Separate the numerical and categorical data
    numerical = [col for col in df.columns if df[col].dtype != 'object']
    categorical = [col for col in df.columns if df[col].dtype == 'object']

    # Apply the outliers removal
    df_diabet = remove_outliers(df_diabet, numerical)
    df_ndiabet = remove_outliers(df_ndiabet, numerical)

    # Concat the splitted dataframe to the original dataframe
    df = pd.concat([df_diabet, df_ndiabet])
    if df.empty:
        raise ValueError('Dataset is empty after outliers handling.')

    # Encode all the categorical features
    # with pd.option_context('future.no_silent_downcasting', True):
    df.replace({
        'gender': {'Male': 0, 'Female': 1, 'Other': 2},
        'race': {
            'AfricanAmerican': 0, 'Asian': 1,
            'Caucasian': 2, 'Hispanic': 3, 'Other': 4,
        },
        'location': {
            'Alabama': 0, 'Alaska': 1, 'Arizona': 2, 'Arkansas': 3,
            'California': 4, 'Colorado': 5, 'Connecticut': 6, 'Delaware': 7,
            'District of Columbia': 8, 'Florida': 9, 'Georgia': 10, 'Guam': 11,
            'Hawaii': 12, 'Idaho': 13, 'Illinois': 14, 'Indiana': 15, 'Iowa': 16,
            'Kansas': 17, 'Kentucky': 18, 'Louisiana': 19, 'Maine': 20,
            'Maryland': 21, 'Massachusetts': 22, 'Michigan': 23, 'Minnesota': 24,
            'Mississippi': 25, 'Missouri': 26, 'Montana': 27, 'Nebraska': 28,
            'Nevada': 29, 'New Hampshire': 30, 'New Jersey': 31, 'New Mexico': 32,
            'New York': 33, 'North Carolina': 34, 'North Dakota': 35, 'Ohio': 36,
            'Oklahoma': 37, 'Oregon': 38, 'Pennsylvania': 39, 'Puerto Rico': 40,
            'Rhode Island': 41, 'South Carolina': 42, 'South Dakota': 43,
            'Tennessee': 44, 'Texas': 45, 'United States': 46, 'Utah': 47,
            'Vermont': 48, 'Virgin Islands': 49, 'Virginia': 50,
            'Washington': 51, 'West Virginia': 52, 'Wisconsin': 53,
        },
        'smoking_history': {
            'No Info': 0, 'never': 1, 'ever': 2,
            'current': 3, 'not current': 4, 'former': 5,
        },
    }, inplace=True)

    # Split the independent variable (X) and dependent variable/label (y)
    # Drop year and race features as they don't give strong correlation to diabetes
    X = df.drop(['diabetes', 'year', 'race'], axis=1)
    y = df['diabetes']

    # Apply the over-undersampling method to overcome the imbalanced data
    smoteenn = SMOTEENN(sampling_strategy='all', random_state=20250531)
    X_smoteenn, y_smoteenn = smoteenn.fit_resample(X, y)

    # Train-test split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_smoteenn, y_smoteenn, test_size=0.2, random_state=20250531, stratify=y_smoteenn
    )

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the cleaned data to a csv file
    preproc_data = pd.DataFrame(X_train, columns=X.columns.tolist())
    preproc_data['diabetes'] = y_train.reset_index(drop=True)
    preproc_data.to_csv(output_path, index=False)
    
    print(f'Pre-processed data successfully saved to: {output_path}\n')
    preproc_data.info()

    return X_train, X_test, y_train, y_test