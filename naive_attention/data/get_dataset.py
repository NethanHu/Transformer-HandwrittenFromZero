from pathlib import Path

import pandas as pd
from datasets import load_dataset

data_file = Path("./naive_attention/data/tinystories.csv")

if not data_file.exists():
    # 1. Loading datasets from Hugging Face Hub
    print("Downloading TinyStories dataset (taking some minutes)...")
    dataset = load_dataset("roneneldan/TinyStories", split='train[:50000]')
    print("Download completed!")

    # 2. Hugging Face to Pandas DataFrame
    df = dataset.to_pandas()

    # 3. We only keep 'text' column
    df_single_column = df[['text']]

    # 4. Saved into single csv file
    df_single_column.to_csv(data_file, index=False, encoding='utf-8')

else:
    print(f"Reading file: {data_file}")
    print(f"Size of the file: {data_file.stat().st_size / (1024*1024):.2f} MB")
    df = pd.read_csv(data_file)
    print(f"  - Rows #: {len(df):,}")
    print(f"  - Cols #: {len(df.columns)}")
    print(f"  - Col Name: {list(df.columns)}")
    print(df.head(10))

