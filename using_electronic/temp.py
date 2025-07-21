import pandas as pd
from tabulate import tabulate

def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df


if __name__ == '__main__':
    train_path = 'datas/train.csv'

    train_df = load_data(train_path)
    print(tabulate(train_df.head(), headers=train_df.columns))