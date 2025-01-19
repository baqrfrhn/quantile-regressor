import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import ta
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AggressivePredictor:
    def __init__(self, pred_days=5, confidence=0.95):
        self.pred_days = pred_days
        self.confidence = confidence
        self.feature_scaler = RobustScaler()
        self.return_scaler = RobustScaler()
        
        # Asset class tickers (rest of the initialization code remains the same)
        self.ASSETS = {
            'SPX': '^GSPC',    # S&P 500
            'NDX': '^NDX',     # Nasdaq 100
            'RTY': '^RUT',     # Russell 2000
            'VIX': '^VIX',     # VIX
            'VVIX': '^VVIX',   # VIX of VIX
            'SKEW': '^SKEW',   # SKEW Index
            'TNX': '^TNX',     # 10Y Yield
            'TYX': '^TYX',     # 30Y Yield
            'DXY': 'DX-Y.NYB', # Dollar Index
            'GC': 'GC=F',      # Gold
            'CL': 'CL=F',      # Oil
            'SECTORS': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE']
        }

    def get_market_data(self, start_date='2010-01-01'):
        """Get comprehensive market data across asset classes."""
        print("Downloading market data...")
        data = {}
        
        # Download main indices
        for name, ticker in self.ASSETS.items():
            if name != 'SECTORS':
                try:
                    data[name] = yf.download(ticker, start=start_date, progress=False)
                except Exception as e:
                    print(f"Failed to download {name}: {e}")
        
        # Download sectors
        for sector in self.ASSETS['SECTORS']:
            try:
                data[sector] = yf.download(sector, start=start_date, progress=False)
            except Exception as e:
                print(f"Failed to download {sector}: {e}")
        
        return data
        
    def calculate_features(self, market_data):
        """Calculate extensive feature set."""
        print("Calculating advanced features...")
        df = pd.DataFrame()
        spx = market_data['SPX']
        
        # 1. Multi-timeframe returns and volatility
        windows = [1, 5, 20, 15, 30, 50]
        for window in windows:
            # Price momentum
            df[f'Return_{window}d'] = spx['Close'].pct_change(window)
            # Volatility
            df[f'Vol_{window}d'] = df[f'Return_{window}d'].rolling(window).std() * np.sqrt(252)
            # Volume trends
            df[f'Volume_MA_{window}'] = spx['Volume'].rolling(window).mean()
            df[f'Volume_Ratio_{window}'] = spx['Volume'] / df[f'Volume_MA_{window}']
            # Price levels
            df[f'MA_{window}'] = spx['Close'].rolling(window).mean()
            df[f'Price_Distance_{window}'] = spx['Close'] / df[f'MA_{window}'] - 1
            
            # Higher moments of returns
            returns = df[f'Return_{window}d']
            df[f'Skew_{window}d'] = returns.rolling(window).skew()
            df[f'Kurt_{window}d'] = returns.rolling(window).kurt()
            
            # Drawdown measures
            rolling_max = spx['Close'].rolling(window).max()
            df[f'Drawdown_{window}d'] = (spx['Close'] - rolling_max) / rolling_max
        
        # 2. Technical indicators
        print("Calculating technical indicators...")
        # Volume indicators
        df['MFI'] = ta.volume.money_flow_index(spx['High'], spx['Low'], spx['Close'], spx['Volume'])
        df['ADI'] = ta.volume.acc_dist_index(spx['High'], spx['Low'], spx['Close'], spx['Volume'])
        df['OBV'] = ta.volume.on_balance_volume(spx['Close'], spx['Volume'])
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(spx['Close'])
        df['Stoch'] = ta.momentum.stoch(spx['High'], spx['Low'], spx['Close'])
        df['Stoch_Signal'] = ta.momentum.stoch_signal(spx['High'], spx['Low'], spx['Close'])
        df['WILLR'] = ta.momentum.williams_r(spx['High'], spx['Low'], spx['Close'])
        
        # Trend indicators
        macd = ta.trend.MACD(spx['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['ADX'] = ta.trend.adx(spx['High'], spx['Low'], spx['Close'])
        df['CCI'] = ta.trend.cci(spx['High'], spx['Low'], spx['Close'])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(spx['Close'])
        df['BB_Width'] = bb.bollinger_wband()
        
        # 3. Cross-asset relationships
        print("Calculating cross-asset features...")
        base_returns = spx['Close'].pct_change()
        
        # Correlations with other assets
        for asset, data in market_data.items():
            if asset not in ['SPX', 'SECTORS']:
                asset_returns = data['Close'].pct_change()
                # Rolling correlations
                for window in [10, 20, 50]:
                    df[f'Corr_{asset}_{window}d'] = base_returns.rolling(window).corr(asset_returns)
                # Relative strength
                df[f'RS_{asset}'] = (data['Close'] / data['Close'].shift(20)) / (spx['Close'] / spx['Close'].shift(20))
        
        # 4. Market regime indicators
        print("Calculating regime indicators...")
        # Volatility regime
        df['VIX'] = market_data['VIX']['Close']
        df['VIX_MA10'] = df['VIX'].rolling(10).mean()
        df['VIX_Regime'] = pd.qcut(df['VIX'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['VVIX_Ratio'] = market_data['VVIX']['Close'] / df['VIX']
        
        # Trend regime
        df['Trend_Regime'] = 'Sideways'
        df.loc[df['Price_Distance_50'] > 0.05, 'Trend_Regime'] = 'Bull'
        df.loc[df['Price_Distance_50'] < -0.05, 'Trend_Regime'] = 'Bear'
        
        # Convert categorical variables to dummy variables
        regime_dummies = pd.get_dummies(df[['VIX_Regime', 'Trend_Regime']], prefix=['VIX', 'Trend'])
        df = pd.concat([df.drop(['VIX_Regime', 'Trend_Regime'], axis=1), regime_dummies], axis=1)
        
        # 5. Target variable
        df['Target_Return'] = df['Return_1d'].rolling(self.pred_days).sum().shift(-self.pred_days)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Final feature count: {len(df.columns)}")
        df.head(10)
        return df

    def train_model(self, X_train, y_train, quantile):
        """Train a heavy quantile regression model."""
        return GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=2000,     # More trees
            max_depth=6,           # Deeper trees
            learning_rate=0.01,    # Slower learning rate
            subsample=0.8,
            min_samples_leaf=10,   # Prevent overfitting
            random_state=42
        ).fit(X_train, y_train.ravel())
        
    def train_and_predict(self):
        """Train model suite and generate predictions."""
        # Get and prepare data
        market_data = self.get_market_data()
        df = self.calculate_features(market_data)
        
        # Prepare features and target
        features = [col for col in df.columns if col != 'Target_Return']
        X = df[features]
        y = df['Target_Return']
        
        # Scale data
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.return_scaler.fit_transform(y.values.reshape(-1, 1))
        
        # Split with gap
        train_size = int(len(X_scaled) * 0.8)
        gap = self.pred_days * 2
        
        X_train = X_scaled[:train_size-gap]
        X_test = X_scaled[train_size:]
        y_train = y_scaled[:train_size-gap]
        y_test = y_scaled[train_size:]
        
        # Train models
        print("\nTraining quantile models...")
        quantiles = [0.05, 0.5, 0.95]
        models = {}
        for q in quantiles:
            print('quantile: ', q)
            models[q] = self.train_model(X_train, y_train, q)
        
        # Generate predictions
        predictions = {q: models[q].predict(X_test) for q in quantiles}
        
        # Calculate intervals
        median = predictions[0.5]
        lower = predictions[0.05]
        upper = predictions[0.95]
        
        # Transform back to original scale
        median = self.return_scaler.inverse_transform(median.reshape(-1, 1)).ravel()
        lower = self.return_scaler.inverse_transform(lower.reshape(-1, 1)).ravel()
        upper = self.return_scaler.inverse_transform(upper.reshape(-1, 1)).ravel()
        y_test_orig = self.return_scaler.inverse_transform(y_test).ravel()
        
        # Calculate metrics
        coverage = np.mean((y_test_orig >= lower) & (y_test_orig <= upper))
        rmse = np.sqrt(np.mean((y_test_orig - median) ** 2))
        avg_interval = np.mean(upper - lower)
        
        print("\nPerformance Metrics:")
        print(f"Coverage: {coverage:.2%} (target: {self.confidence:.2%})")
        print(f"RMSE: {rmse:.4f}")
        print(f"Average Interval Width: {avg_interval:.4f}")
        
        # Plot results
        plt.figure(figsize=(20, 10))
        
        dates = df.index[-len(y_test_orig):]
        plt.plot(dates, y_test_orig, label='Actual Returns', color='blue', alpha=0.6)
        plt.plot(dates, median, label='Median Prediction', color='red', alpha=0.7)
        plt.fill_between(dates, lower, upper, color='red', alpha=0.2, 
                        label=f'{self.confidence*100}% Confidence Interval')
        
        # Add regime overlay
        vol = df['Vol_20d'].iloc[-len(y_test_orig):]
        plt.fill_between(dates, plt.ylim()[0], plt.ylim()[1], 
                        where=vol > vol.mean() + vol.std(),
                        color='gray', alpha=0.1, label='High Volatility Regime')
        
        plt.title(f'{self.pred_days}-Day Return Prediction with Market Regimes')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add volatility subplot
        ax2 = plt.gca().twinx()
        ax2.plot(dates, vol * np.sqrt(252), 
                color='purple', alpha=0.3, label='Annualized Volatility')
        ax2.set_ylabel('Annualized Volatility', color='purple')
        
        plt.tight_layout()
        plt.show()
        
        # Plot feature importance
        importance = pd.Series(
            models[0.5].feature_importances_,
            index=features
        ).sort_values(ascending=True)
        
        plt.figure(figsize=(12, 8))
        importance.tail(20).plot(kind='barh')
        plt.title('Top 20 Important Features')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
        
        return models, coverage, rmse

    def save_model(self, models):
        """Save models and scalers."""
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            
        # Save models
        for quantile, model in models.items():
            with open(f'saved_models/model_{quantile}.pkl', 'wb') as f:
                pickle.dump(model, f)
                
        # Save scalers
        with open('saved_models/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open('saved_models/return_scaler.pkl', 'wb') as f:
            pickle.dump(self.return_scaler, f)
            
        print("Models and scalers saved successfully!")
        
    def load_model(self):
        """Load saved models and scalers."""
        if not os.path.exists('saved_models'):
            print("No saved models found.")
            return None
            
        try:
            # Load models
            models = {}
            for quantile in [0.05, 0.5, 0.95]:
                with open(f'saved_models/model_{quantile}.pkl', 'rb') as f:
                    models[quantile] = pickle.load(f)
                    
            # Load scalers
            with open('saved_models/feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            with open('saved_models/return_scaler.pkl', 'rb') as f:
                self.return_scaler = pickle.load(f)
                
            print("Models and scalers loaded successfully!")
            return models
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

if __name__ == "__main__":
    predictor = AggressivePredictor(pred_days=5, confidence=0.95)
    
    # Try to load existing models
    models = predictor.load_model() 
    
    if models is None:
        print("Training new models...")
        models, coverage, rmse = predictor.train_and_predict()
        # Save the models after training
        predictor.save_model(models)
    else:
        # Use loaded models to make predictions
        market_data = predictor.get_market_data()
        df = predictor.calculate_features(market_data)
        
        features = [col for col in df.columns if col != 'Target_Return']
        X = df[features]
        y = df['Target_Return']
        
        X_scaled = predictor.feature_scaler.transform(X)
        y_scaled = predictor.return_scaler.transform(y.values.reshape(-1, 1))
        
        train_size = int(len(X_scaled) * 0.8)
        gap = predictor.pred_days * 2
        X_test = X_scaled[train_size:]
        y_test = y_scaled[train_size:]
        
        predictions = {q: models[q].predict(X_test) for q in models.keys()}
        
        median = predictions[0.5]
        lower = predictions[0.05]
        upper = predictions[0.95]
        
        median = predictor.return_scaler.inverse_transform(median.reshape(-1, 1)).ravel()
        lower = predictor.return_scaler.inverse_transform(lower.reshape(-1, 1)).ravel()
        upper = predictor.return_scaler.inverse_transform(upper.reshape(-1, 1)).ravel()
        y_test_orig = predictor.return_scaler.inverse_transform(y_test).ravel()
        
        coverage = np.mean((y_test_orig >= lower) & (y_test_orig <= upper))
        rmse = np.sqrt(np.mean((y_test_orig - median) ** 2))
        avg_interval = np.mean(upper - lower)
        
        print("\nPerformance Metrics:")
        print(f"Coverage: {coverage:.2%} (target: {predictor.confidence:.2%})")
        print(f"RMSE: {rmse:.4f}")
        print(f"Average Interval Width: {avg_interval:.4f}")
        
        # Create visualization
        plt.figure(figsize=(20, 10))
        
        dates = df.index[-len(y_test_orig):]
        plt.plot(dates, y_test_orig, label='Actual Returns', color='blue', alpha=0.6)
        plt.plot(dates, median, label='Median Prediction', color='red', alpha=0.7)
        plt.fill_between(dates, lower, upper, color='red', alpha=0.2, 
                        label=f'{predictor.confidence*100}% Confidence Interval')
        
        vol = df['Vol_20d'].iloc[-len(y_test_orig):]
        plt.fill_between(dates, plt.ylim()[0], plt.ylim()[1], 
                        where=vol > vol.mean() + vol.std(),
                        color='gray', alpha=0.1, label='High Volatility Regime')
        
        plt.title(f'{predictor.pred_days}-Day Return Prediction with Market Regimes')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        ax2 = plt.gca().twinx()
        ax2.plot(dates, vol * np.sqrt(252), 
                color='purple', alpha=0.3, label='Annualized Volatility')
        ax2.set_ylabel('Annualized Volatility', color='purple')
        
        plt.tight_layout()
        plt.show()