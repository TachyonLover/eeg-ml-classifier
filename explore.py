from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

dataset = fetch_ucirepo(id=264)

x = dataset.data.features
y = dataset.data.targets

print("Features:")
print(x.head())

print("\nTargets:")
print(y.head())

print("\nDataset shape:")
print(x.shape)

print("\nColumn names:")
print(x.columns)

print("\nData types:")
print(x.dtypes)

print("\nSummary statistics:")
print(x.describe())

print("\nEye state distribution:")
print(y.value_counts())

print("\nMissing values:")
print(x.isnull().sum())

df = x.copy()
df["eyeDetection"] = y["eyeDetection"]

averages = df.groupby("eyeDetection").mean()

print("\nAverage EEG measurements by eye state:")
print(averages)

differences = abs(averages.loc[1] - averages.loc[0])

print(differences.sort_values(ascending=False))

print("\nAF4 statistics:")
print(df["AF4"].describe())

print("\nLargest AF4 measurements:")
print(df["AF4"].nlargest(5))

print("\nSmallest AF4 measurements:")
print(df["AF4"].nsmallest(5))

open_eyes = df[df["eyeDetection"] == 0]["AF4"]
closed_eyes = df[df["eyeDetection"] == 1]["AF4"]

plt.hist(open_eyes, bins=50, range=(3500, 5500),
         alpha=0.5, label="Eyes Open")

plt.hist(closed_eyes, bins=50, range=(3500, 5500),
         alpha=0.5, label="Eyes Closed")

plt.xlabel("AF4 EEG Measurement")
plt.ylabel("Number of Observations")
plt.title("AF4 Distribution by Eye State")
plt.legend()

plt.show()

print("\nMinimum values:")
print(x.min())

print("\nMaximum values:")
print(x.max())

print("\n99th percentile:")
print(x.quantile(0.99))

outlier_index = x["AF4"].idxmax()

print("\nObservation with largest AF4:")
print(df.loc[outlier_index])