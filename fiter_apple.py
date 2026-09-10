import pandas as pd

input_file = "twcs.csv"
output_file = "apple_support.csv"

apple_rows = []

for chunk in pd.read_csv(input_file, chunksize=100000):
    mask = chunk["text"].astype(str).str.contains(
        "@AppleSupport",
        case=False,
        na=False
    )

    apple_rows.append(chunk[mask])

apple_df = pd.concat(apple_rows, ignore_index=True)

apple_df.to_csv(output_file, index=False)

print("Apple tweets:", len(apple_df))
print("Saved to:", output_file)