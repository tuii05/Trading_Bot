## Trading Bot in Python

The trading bot can be tested by running `Test.py`.

When initializing the trading bot, two key parameters should be provided:
- `Stock`: the ticker symbol, supported by Yahoo Finance API.
- `Model`: the name of the prediction model to be used.

The trading bot has three models implemented in it:

- `RFC` - A basic Random Forest Classifier implemented using Scikit-learn.
- `XGB` - A gradient boosting predictive model from the XGBoost library.
- `Merton` - A custom implementation of the Merton Jump Diffusion Model, optimized with multithreading for improved performance.

Notes on the Merton Jump Diffusion model:

- The initial parameter (e.g. volatility, jump intensity) of the model should be manually tuned for each stock to achieve better anual returns.
- Dividend yield is set to zero by default. For dividend-paying stocks, this must be adjusted accordingly to reflect more accurate pricing dynamics.


