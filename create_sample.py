import pandas as pd

df = pd.read_csv("medium_articles_sample.csv")

df_clean = df.dropna(subset=['title', 'text'])

df_sample = df_clean.sample(n=10, random_state=42)

output_filename = "medium_articles_sample_test.csv"
df_sample.to_csv(output_filename, index=False)

print(f"Successfully saved {len(df_sample)} articles to {output_filename}.")