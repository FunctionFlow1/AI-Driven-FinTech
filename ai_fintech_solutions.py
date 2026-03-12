import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class AlgorithmicTrading:
    """Implements various algorithmic trading strategies."""
    def sma_crossover_strategy(self, ticker, start_date, end_date, short_window=40, long_window=100):
        """Executes a Simple Moving Average (SMA) Crossover strategy."""
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data downloaded for {ticker}. Check ticker or date range.")
            return pd.DataFrame()

        data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
        data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()
        data["Signal"] = 0
        # Generate signal: 1 for buy, 0 for hold/sell
        data["Signal"][short_window:] = np.where(data["SMA_Short"][short_window:] > data["SMA_Long"][short_window:], 1, 0)
        data["Position"] = data["Signal"].diff()
        
        # Backtesting logic (simplified for demonstration)
        initial_capital = 100000.0
        positions = pd.DataFrame(index=data.index).fillna(0.0)
        positions[ticker] = 100 * data["Signal"]
        
        portfolio = pd.DataFrame(index=data.index)
        portfolio["Holdings"] = (positions.multiply(data["Adj Close"], axis=0)).sum(axis=1)
        
        # Calculate cash changes based on position changes
        pos_diff = positions.diff()
        trade_costs = (pos_diff.multiply(data["Adj Close"], axis=0)).sum(axis=1)
        portfolio["Cash"] = initial_capital - trade_costs.cumsum().fillna(0)
        
        portfolio["Total"] = portfolio["Cash"] + portfolio["Holdings"]
        portfolio["Returns"] = portfolio["Total"].pct_change()
        
        print(f"--- {ticker} Trading Strategy Results ---")
        print(portfolio.tail())
        return portfolio

class FraudDetection:
    """Provides tools for building and evaluating fraud detection models."""
    def train_fraud_detection_model(self, X, y):
        """Trains a RandomForestClassifier for fraud detection."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=\"balanced\")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("--- Fraud Detection Model Report ---")
        print(classification_report(y_test, y_pred))
        return model

# Main execution block for demonstration
if __name__ == "__main__":
    # Algorithmic Trading Example
    trading_bot = AlgorithmicTrading()
    apple_portfolio = trading_bot.sma_crossover_strategy("AAPL", "2020-01-01", "2023-01-01")
    print("\n" + "="*50 + "\n")

    # Fraud Detection Example (dummy data)
    print("Generating dummy data for fraud detection...")
    np.random.seed(42)
    num_transactions = 1000
    data = {
        "transaction_amount": np.random.rand(num_transactions) * 1000 + 50,
        "transaction_frequency_30d": np.random.randint(1, 50, num_transactions),
        "merchant_risk_score": np.random.rand(num_transactions) * 10,
        "is_fraud": np.random.choice([0, 1], num_transactions, p=[0.98, 0.02]) # 2% fraud
    }
    transactions_df = pd.DataFrame(data)

    # Introduce some patterns for fraud
    transactions_df.loc[transactions_df["is_fraud"] == 1, "transaction_amount"] = np.random.rand(transactions_df["is_fraud"].sum()) * 2000 + 1000
    transactions_df.loc[transactions_df["is_fraud"] == 1, "merchant_risk_score"] = np.random.rand(transactions_df["is_fraud"].sum()) * 5 + 5

    X = transactions_df[["transaction_amount", "transaction_frequency_30d", "merchant_risk_score"]]
    y = transactions_df["is_fraud"]

    fraud_detector = FraudDetection()
    fraud_model = fraud_detector.train_fraud_detection_model(X, y)
    print("\nFraud detection model trained successfully.")

    # Example of using the trained model to predict new data
    new_transaction = pd.DataFrame([{
        "transaction_amount": 1500,
        "transaction_frequency_30d": 5,
        "merchant_risk_score": 8
    }])
    prediction = fraud_model.predict(new_transaction)
    print(f"\nPrediction for new transaction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
