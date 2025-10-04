import pandas as pd

# Load the CSV files
true_news = pd.read_csv('dataset/true.csv')
fake_news = pd.read_csv('dataset/fake.csv')

# Add labels: 0 = Real, 1 = Fake
true_news['label'] = 0
fake_news['label'] = 1

# Combine both datasets
combined = pd.concat([true_news, fake_news], ignore_index=True)

# Keep only 'text' and 'label' columns (adjust 'text' if your column name is different)
combined = combined[['text', 'label']]

# Shuffle the data
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as a new CSV
combined.to_csv('dataset/fake_or_real_news.csv', index=False)

print("Combined dataset saved as dataset/fake_or_real_news.csv")

