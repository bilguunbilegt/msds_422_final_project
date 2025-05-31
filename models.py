import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

# 2. Feature Selection and Preprocessing
def preprocess_data(df, target_col='AirQualityIndex'):
    X = df.drop(columns=[target_col, 'Date', 'Time'])  # Drop date/time
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# 3. Train Models
def train_models(X_train, y_train, X_val, y_val):
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['LinearRegression'] = lr

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # K-Nearest Neighbors
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    # Neural Network
    nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    nn.fit(X_train, y_train, validation_data=(X_val, y_val),
           epochs=100, batch_size=16, verbose=0,
           callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
    models['NeuralNetwork'] = nn

    return models

# 4. Evaluate and Save
def evaluate_models(models, X_val, y_val, scaler, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for name, model in models.items():
        if name == 'NeuralNetwork':
            y_pred = model.predict(X_val).flatten()
        else:
            y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        results.append({'Model': name, 'R2': r2, 'RMSE': rmse})

        # Save models
        if name == 'NeuralNetwork':
            model.save(os.path.join(output_dir, f'{name}.h5'))
        else:
            joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

# 5. Visualize
def plot_data(df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Daily Trends of Pollutants
    pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'O3(GT)', 'PM2.5', 'PM10']
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('DateTime', inplace=True)

    df[pollutants].resample('D').mean().plot(figsize=(12, 6))
    plt.title('Daily Average Pollutants Over Time')
    plt.ylabel('Concentration')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_pollutants.png'))
    plt.close()

# Main
if __name__ == "__main__":
    df = load_data('AirQualityData.csv')
    plot_data(df)
    X, y, scaler = preprocess_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train, X_val, y_val)
    evaluate_models(models, X_val, y_val, scaler)
