import pandas as pd

df = pd.read_csv("itctray3.csv")
df1 = df.groupby("author")["counts"]
print(df1.head(5))
