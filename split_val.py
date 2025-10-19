
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train_all.csv")
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)