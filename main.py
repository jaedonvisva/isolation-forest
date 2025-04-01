import pandas as pd
import matplotlib.pyplot as plt
import isolation_forest

df_building_metadata = pd.read_csv('data/building_metadata.csv')
df_weather = pd.read_csv('data/weather_train.csv')
df_train = pd.read_csv('data/train.csv')

df_train = df_train.merge(df_building_metadata, on='building_id', how='left')
df_train = df_train[df_train['meter_reading'] > 0]
df_train  = df_train.drop(columns=["building_id", "timestamp", "meter", "site_id", "year_built", "floor_count", "primary_use"])

x = df_train["square_feet"].head(40)
y = df_train["meter_reading"].head(40)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter plot of Square Feet vs Meter Reading')
plt.xlabel('Square Feet')
plt.ylabel('Meter Reading')

np_train = df_train.head(40).to_numpy()
forest = isolation_forest.isolation_forest(np_train, n_trees=300, max_samples=256)
psi = min(256, len(np_train))
scores = isolation_forest.compute_scores(forest, np_train, psi)

anomaly_mask = scores > 0.6  

x_anomalies = x[anomaly_mask]
y_anomalies = y[anomaly_mask]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Normal Data')

plt.scatter(x_anomalies, y_anomalies, color='red', label='Anomalies')

plt.title('Scatter plot of Square Feet vs Meter Reading')
plt.xlabel('Square Feet')
plt.ylabel('Meter Reading')
plt.legend()
plt.show()

