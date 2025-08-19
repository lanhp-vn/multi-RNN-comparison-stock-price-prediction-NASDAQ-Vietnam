# Time-series Data And Application To Stock Market

A comprehensive deep learning project implementing stock price prediction and portfolio optimization using RNN models for both NASDAQ and Vietnam stock markets.

## Project Overview

This project explores time-series analysis and predictive modeling for stock market data, implementing four main tasks:

1. **NASDAQ Stock Price Prediction** - Price forecasting using RNN models
2. **Vietnam Stock Price Prediction** - Price forecasting for Vietnamese stocks
3. **Vietnam Trading Point Identification** - Buy/sell signal generation
4. **NASDAQ Portfolio Management** - Automated portfolio optimization

## Key Results

### NASDAQ Stock Price Prediction (MSFT)
- **Bi-LSTM Model**: MSE 0.034 (30 days), MSE 0.0028 (252 days)
- **GRU Model**: MSE 0.0337 (30 days), MSE 0.0028 (252 days)
- **Conv1D Model**: MSE 0.06 (30 days), MSE 0.035 (252 days)

### Vietnam Stock Price Prediction (VNA)
- **Bi-LSTM Model**: MSE 0.015 (50 days), MSE 0.0038 (252 days)
- **GRU Model**: MSE 0.015 (50 days), MSE 0.0038 (252 days)
- **Conv1D Model**: MSE 0.069 (50 days), MSE 0.18 (252 days)

### NASDAQ Portfolio Performance (Industrials Sector)
- **Final MSE**: 0.0023
- **6 out of 7 companies** recommended to HOLD
- **1 company (PCAR)** recommended to GET RID
- **Best performer**: CSX with potential score 8.64 and profit 5.19

## Project Structure

```
Final Project DL4AI/
├── 1.NASDAQStockPricePrediction.ipynb    # NASDAQ price prediction models
├── 2.VietnamStockPricePrediction.ipynb   # Vietnam price prediction models
├── 3.VietnamTradingPointIdentification.ipynb  # Trading signals using SMA/EMA
├── 4.NASDAQPortfolio.ipynb               # Portfolio optimization system
├── MSFT.csv                              # Microsoft stock data
├── VNA-UP~1.CSV                         # Vietnam Airlines stock data
├── nasdaq_100_data.csv                   # NASDAQ 100 companies data
├── nasdaq_100_data_withIndustry.csv      # NASDAQ data with industry classification
├── nasdaq_data_preprocessed.csv          # Preprocessed NASDAQ data
├── portfolio.csv                         # Final portfolio recommendations
├── nasdaq_preprocessed/                  # Individual company CSV files
│   ├── CPRT.csv
│   ├── CSX.csv
│   ├── CTAS.csv
│   ├── FAST.csv
│   ├── PCAR.csv
│   ├── VRSK.csv
│   └── VRSN.csv
├── trade_tables/                         # Generated trading tables
│   ├── CPRT_trade_table.csv
│   ├── CSX_trade_table.csv
│   ├── CTAS_trade_table.csv
│   ├── FAST_trade_table.csv
│   ├── PCAR_trade_table.csv
│   ├── VRSK_trade_table.csv
│   └── VRSN_trade_table.csv
├── project-requirements.pdf             # Original project requirements
├── project-report.pdf                   # Technical report
└── README.md                            # This file
```

## Models & Techniques

### Deep Learning Models
- **Bidirectional LSTM**: Best overall performance for long-term predictions
- **GRU**: Excellent performance with faster training
- **1D Convolution**: Moderate performance, not optimal for stock data

### Technical Indicators
- **Simple Moving Average (SMA)**: 50-day and 100-day crossover strategy
- **Exponential Moving Average (EMA)**: 50-day and 100-day crossover strategy

### Key Hyperparameters
- **NASDAQ Models**: 200 epochs, batch size 4096, 10-fold cross-validation
- **Vietnam Models**: 150 epochs, batch size 2048, 5-fold cross-validation
- **Window Sizes**: 30/50 days (short-term), 252 days (long-term)

## Key Findings

1. **Larger window sizes (252 days) consistently outperform smaller windows (30/50 days)** across all models
2. **Bi-LSTM and GRU models show similar excellent performance** for stock price prediction
3. **Conv1D models are not suitable for stock price prediction** due to high MSE values
4. **Cross-validation works better for smaller window sizes** in both markets
5. **Industrial sector stocks show high stability** with 6/7 companies profitable
6. **SMA and EMA crossover strategies effectively identify trading points** with smaller windows providing more responsive signals

## Portfolio Recommendations

| Company | Profit | Potential Score | Risk Score | Action |
|---------|--------|----------------|------------|--------|
| CSX | 5.19 | 8.64 | 5.18 | **Hold** |
| VRSK | 2.86 | 4.77 | 34.33 | **Hold** |
| CTAS | 2.07 | 3.45 | 92.32 | **Hold** |
| FAST | 2.19 | 3.64 | 11.06 | **Hold** |
| VRSN | 2.33 | 3.87 | 31.45 | **Hold** |
| CPRT | 0.10 | 0.17 | 15.98 | **Hold** |
| PCAR | -0.60 | -1.00 | 8.14 | **Get Rid** |

## Technology Stack

### Core Libraries
- **TensorFlow/Keras**: Deep learning model implementation
- **scikit-learn**: Data preprocessing and evaluation metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization

### Data Sources
- **NASDAQ**: Microsoft (MSFT) historical data
- **Vietnam**: Vietnam Airlines (VNA) historical data
- **Portfolio**: NASDAQ-100 Industrial sector companies

## Results Interpretation

### Model Performance
- **MSE < 0.01**: Excellent prediction accuracy
- **MSE 0.01-0.05**: Good prediction accuracy
- **MSE > 0.05**: Poor prediction accuracy (Conv1D models)

### Trading Signals
- **SMA/EMA Crossover**: When short-term average crosses above long-term = BUY signal
- **SMA/EMA Crossover**: When short-term average crosses below long-term = SELL signal

### Portfolio Metrics
- **Potential Score**: Profit normalized by worst performer (higher = better)
- **Risk Score**: Standard deviation of closing prices (lower = less volatile)
- **Action**: Hold (profitable) vs Get Rid (unprofitable)

## Future Improvements

1. **Enhanced Features**: Include volume, technical indicators (RSI, MACD, Bollinger Bands)
2. **Multi-Asset Models**: Combine multiple stocks for better market understanding
3. **Risk Management**: Implement stop-loss and take-profit mechanisms
4. **Real-time Deployment**: Create API service for live trading signals
5. **Alternative Models**: Explore Transformer architectures and ensemble methods

## Academic Context

This project was completed as part of CS209 Deep Learning for Artificial Intelligence course at Fulbright University Vietnam, Spring 2023. The work demonstrates practical application of deep learning techniques to financial time-series analysis and automated trading systems.

## Author

**Pham Hoang Lan (UG190042)**  
Fulbright University Vietnam  
Instructor: Prof. Dang Huynh  
May 26th, 2023

## Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain, and the models presented should not be used for actual financial trading without proper risk management and professional advice.
