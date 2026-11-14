

import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


DATA_PATH = 'data/student-por.csv'
OUTPUT_FILE = 'model.bin'
ALPHA = 10

numerical = [
    'age', 'medu', 'fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'absences',
]

categorical = [
    'school', 'sex', 'address', 'famsize', 'pstatus',
    'mjob', 'fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic',
]


def load_data(path):
    df = pd.read_csv(path, sep=';')

    df.columns = df.columns.str.lower()

    string_columns = list(df.select_dtypes(exclude='number').columns)

    for c in string_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # students with G3 = 0 are likely dropouts, see notebook
    df = df[df.g3 > 0].reset_index(drop=True)

    # G1 and G2 are not known at the start of the school year
    df = df.drop(columns=['g1', 'g2'])

    return df


def train_model(df, y, alpha):
    dicts = df[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        Ridge(alpha=alpha, random_state=1)
    )
    pipeline.fit(dicts, y)

    return pipeline


def evaluate(pipeline, df, y):
    dicts = df[categorical + numerical].to_dict(orient='records')
    y_pred = pipeline.predict(dicts)

    return root_mean_squared_error(y, y_pred)


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


df = load_data(DATA_PATH)

# sanity check: same split as in the notebook, should give test RMSE 2.137
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

pipeline = train_model(df_full_train, df_full_train.g3.values, ALPHA)
rmse_test = evaluate(pipeline, df_test, df_test.g3.values)
print(f'Test RMSE: {rmse_test:.3f}')

# final model trained on all data
pipeline = train_model(df, df.g3.values, ALPHA)
save_model(pipeline, OUTPUT_FILE)

print(f'Model saved to {OUTPUT_FILE}')