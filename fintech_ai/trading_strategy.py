import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingStrategy:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = None

    def _generate_features(self, df):
        # Simple moving averages and RSI as features
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df = df.dropna()
        return df

    def calculate_rsi(self, series, window=14):
        diff = series.diff(1).dropna()
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train(self, historical_data_df):
        logging.info("Training trading strategy model...")
        df = self._generate_features(historical_data_df.copy())
        
        # Define target: 1 if price goes up next day, 0 otherwise
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()

        features = ['SMA_10', 'SMA_30', 'RSI', 'Volatility']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        logging.info(f"Model trained. Classification Report:
{classification_report(y_test, predictions)}")

    def predict(self, current_data_df):
        logging.info("Making trading prediction...")
        df = self._generate_features(current_data_df.copy())
        if df.empty:
            return 0 # No prediction if no data
        latest_features = df.iloc[[-1]][['SMA_10', 'SMA_30', 'RSI', 'Volatility']]
        prediction = self.model.predict(latest_features)[0]
        return prediction # 1 for buy, 0 for sell/hold

if __name__ == "__main__":
    # Create dummy historical data
    dates = pd.date_range(start='2023-01-01', periods=100)
    close_prices = np.random.randn(100).cumsum() + 100
    dummy_data = pd.DataFrame({'Date': dates, 'Close': close_prices})
    dummy_data['Date'] = pd.to_datetime(dummy_data['Date'])
    dummy_data = dummy_data.set_index('Date')

    strategy = TradingStrategy()
    strategy.train(dummy_data)

    # Simulate new data for prediction
    new_dates = pd.date_range(start='2023-04-11', periods=5)
    new_close_prices = np.random.randn(5).cumsum() + dummy_data['Close'].iloc[-1]
    new_data = pd.DataFrame({'Date': new_dates, 'Close': new_close_prices})
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data = new_data.set_index('Date')

    prediction = strategy.predict(new_data)
    if prediction == 1:
        print("
Prediction: BUY")
    else:
        print("
Prediction: HOLD/SELL")
