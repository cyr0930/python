import numpy as np
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


def impute():
    csv_data = '''A,B,C,D\n1.0,2.0,3.0,4.0\n5.0,6.0,,8.0\n10.0,11.0,12.0,'''
    df = pd.read_csv(StringIO(csv_data))
    print(df)

    # calculate column mean
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)

    # calculate row mean
    ftr_imr = FunctionTransformer(lambda X: imr.fit_transform(X.T).T, validate=False)
    imputed_data = ftr_imr.fit_transform(df.values)
    print(imputed_data)


def categorical_data_one_hot():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']

    df['size'] = df['size'].map({'XL': 3, 'L': 2, 'M': 1})
    df = pd.get_dummies(df, drop_first=True)
    print(df)


impute()
categorical_data_one_hot()
