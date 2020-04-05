import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def object_creation():
    s = pd.Series([1, np.nan])
    dates = pd.date_range('20130101', periods=2)
    df = pd.DataFrame(np.random.randn(2, 3), index=dates, columns=list('ABC'))
    df2 = pd.DataFrame({'A': pd.Timestamp('20130102'),
                        'B': pd.Series(1, index=list(range(2)), dtype='float32'),
                        'C': np.array([3] * 2, dtype='int32'),
                        'D': pd.Categorical(['test', 'train'])})
    print(df2.dtypes)
    return df


def viewing_data():
    print(df.head())
    print(df.tail(1))
    print(df.index)
    print(df.columns)
    # DataFrame.to_numpy() can be an expensive operation when df has columns with different data types
    print(df.to_numpy())
    print(df.describe())
    print(df.T)
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by='B'))


def selection():
    # Getting
    print(df['A'])  # Selecting a single column. Equivalent to df.A.# selecting via [], which slices the rows
    print(df[:2])   # Selecting via [], which slices the rows
    print(df[:'20130102'])
    # Selection by label
    print(df.loc['20130101'])
    print(df.loc[:, ['A', 'B']])
    # Selection by position
    print(df.iloc[1])
    print(df.iloc[:1, 1:2])
    print(df.iloc[[0, 1], [0, 2]])
    print(df.iat[1, 1])     # For getting fast access to a scalar
    # Boolean indexing
    print(df[df['A'] > 0])
    print(df[df > 0])
    df2 = df.copy()
    df2['D'] = ['one', 'two']
    print(df2[df2['D'].isin(['two'])])
    # Setting
    df.at['20130101', 'A'] = 0
    df.iat[0, 1] = 0
    df.loc[:, 'C'] = np.array([5] * len(df))
    print(df)
    df2 = df.copy()
    df2[df2 > 0] = -df2
    print(df2)


def missing_data():
    # pandas uses np.nan to represent missing data
    df1 = df.reindex(index=df.index[:2], columns=list(df.columns) + ['D'])
    df1.loc[:df.index[0], 'D'] = 1
    print(df1)
    print(df1.dropna(how='any'))
    print(df1.fillna(value=5))
    print(pd.isna(df1))


def operations():
    print(df.mean())    # operations in general exclude missing data
    print(df.mean(1))   # same operation on the other axis
    s = pd.Series([1, np.nan], index=df.index).shift(1)
    print(df)
    print(df.sub(s, axis='index'))
    print(df.apply(np.cumsum))
    print(df.apply(lambda x: x.max() - x.min()))
    print(pd.Series(np.random.randint(0, 7, size=10)).value_counts())   # histogramming
    print(pd.Series(['Aaba', np.nan]).str.lower())


def merge():
    # Adding a column to a DataFrame is relatively fast. However, adding a row requires a copy, and may be expensive.
    print(pd.concat([df[:1], df[1:]]))
    left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
    right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
    print(pd.merge(left, right, on='key'))  # join, SQL style merges


def grouping():
    df0 = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                        'C': np.random.randn(8),
                        'D': np.random.randn(8)})
    print(df0.groupby(['A', 'B']).sum())


def reshaping():
    tuples = list(zip(*[['bar', 'bar', 'baz', 'baz'], ['one', 'two', 'one', 'two']]))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df0 = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['A', 'B'])
    print(df0)
    stacked = df0.stack()
    print(stacked)
    print(stacked.unstack())    # by default unstacks the last level
    print(stacked.unstack(0))
    df0 = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                        'B': ['A', 'B', 'C'] * 4,
                        'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                        'D': np.random.randn(12),
                        'E': np.random.randn(12)})
    print(pd.pivot_table(df0, values='D', index=['A', 'B'], columns=['C']))


def categoricals():
    df0 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
    df0["grade"] = df0["raw_grade"].astype("category")    # convert to a categorical data type
    print(df0["grade"])
    df0["grade"].cat.categories = ["very good", "good", "very bad"]  # rename the categories
    print(df0["grade"])
    # Reorder the categories and simultaneously add the missing categories
    df0["grade"] = df0["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
    print(df0["grade"])
    # Sorting is per order in the categories, not lexical order
    print(df0.sort_values(by="grade"))


def plotting():
    index = pd.date_range('1/1/2000', periods=1000)
    df0 = pd.DataFrame(np.random.randn(1000, 4), index=index, columns=['A', 'B', 'C', 'D'])
    df0 = df0.cumsum()
    df0.plot()
    plt.legend(loc='best')
    # plt.show()


def getting_data_in_out():
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    df.to_csv('tmp/foo.csv')
    print(pd.read_csv('tmp/foo.csv'))
    df.to_hdf('tmp/foo.h5', 'df')
    print(pd.read_hdf('tmp/foo.h5', 'df'))
    df.to_excel('tmp/foo.xlsx', sheet_name='Sheet1')
    print(pd.read_excel('tmp/foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']))


df = object_creation()
viewing_data()
selection()
missing_data()
operations()
merge()
reshaping()
categoricals()
plotting()
getting_data_in_out()
