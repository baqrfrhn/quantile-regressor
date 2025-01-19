# Quantile Regression Market Predictor

A sophisticated machine learning system for predicting market returns using quantile regression and comprehensive market analysis across multiple asset classes.

## Overview

This project implements an advanced market prediction system that combines:
- Multi-asset class analysis (equities, volatility indices, rates, commodities)
- Extensive feature engineering across multiple timeframes
- Quantile regression for uncertainty estimation
- Regime detection and adaptive modeling
- Comprehensive technical indicators

## Features

### Data Sources
The system tracks multiple market indices and assets:
- Equity Indices: S&P 500 (SPX), Nasdaq 100 (NDX), Russell 2000 (RTY)
- Volatility: VIX, VVIX, SKEW Index
- Interest Rates: 10Y Yield (TNX), 30Y Yield (TYX)
- Currencies: Dollar Index (DXY)
- Commodities: Gold (GC), Oil (CL)
- All major S&P 500 sectors (XLK, XLF, XLE, etc.)

### Feature Engineering
The system calculates over 100 features including:
- Multi-timeframe returns and volatility (1d, 5d, 15d, 20d, 30d, 50d)
- Volume trends and money flow indicators
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Cross-asset correlations and relative strength
- Market regime indicators
- Higher-moment statistics (skewness, kurtosis)
- Drawdown measures

### Model Architecture
- Uses Gradient Boosting for quantile regression
- Trains three models for different quantiles (0.05, 0.50, 0.95)
- Implements robust scaling for features and returns
- Includes proper train/test splitting with time gaps

## Installation

### Prerequisites
```python
numpy
pandas
yfinance
scikit-learn
matplotlib
ta
pickle
```

### Model Files
The system includes five essential files:
- `feature_scaler.pkl`: Scaler for input features
- `return_scaler.pkl`: Scaler for return values
- `model_0.05.pkl`: Lower bound (5th percentile) model
- `model_0.5.pkl`: Median (50th percentile) model
- `model_0.95.pkl`: Upper bound (95th percentile) model

## Usage

### Basic Usage
```python
from quantileregression import AggressivePredictor

# Initialize predictor
predictor = AggressivePredictor(pred_days=5, confidence=0.95)

# Load existing models
models = predictor.load_model()

# Or train new models
if models is None:
    models, coverage, rmse = predictor.train_and_predict()
    predictor.save_model(models)
```

### Parameters
- `pred_days`: Prediction horizon in trading days (default: 5)
- `confidence`: Confidence interval width (default: 0.95)

## Output

The system provides:
- Median return predictions
- Confidence intervals (default: 95%)
- Performance metrics (coverage ratio, RMSE)
- Visualization of predictions with:
  - Confidence bands
  - Volatility regime overlay
  - Feature importance analysis

## Performance Metrics

The model tracks several key performance indicators:
- Coverage Ratio: Percentage of actual returns falling within predicted intervals
- RMSE: Root Mean Square Error of median predictions
- Average Interval Width: Measure of prediction uncertainty

## Limitations and Considerations

- Requires substantial market data history for training
- Performance depends on market regime stability
- Model assumes some persistence in market relationships
- Should be used as part of a broader analysis framework
- Past performance does not guarantee future results

## Future Improvements

Potential areas for enhancement:
- Integration of alternative data sources
- Dynamic feature selection
- Adaptive training windows
- Online learning capabilities
- Additional asset classes
- Enhanced regime detection

## Contributing

Feel free to submit issues and enhancement requests.