import string
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt
import xgboost as xgb

class tradingBot:
    def __init__(self, stock: string, type: string):
        self.type = type

        if self.type not in ["RFC", "XGB", "Merton"]:
            raise Exception("\nThe following types can be used: RFC, XGB, Merton.")

        self.stock = yf.Ticker(stock)
        self.data = self.stock.history(period="1y")
        self.threshold = 0.002

    def createData(self):
        data = self.data
        data["Volatility"] = data["Close"].rolling(5).std()
        data["Return_1"] = data["Close"].pct_change(1)
        data["Lagged_Return_1"] = data["Return_1"].shift(1)
        data["Lagged_Return_2"] = data["Return_1"].shift(2)
        data["Return_3"] = data["Close"].pct_change(3)
        data["Return_5"] = data["Close"].pct_change(5)
        data["Volume_Delta"] = data["Volume"].pct_change()
        data["Moving_Average"] = data["Close"].rolling(5).mean()
        data["Moving_Average_Slow"] = data["Close"].rolling(10).mean()
        data["Candle_Range"] = data["High"] - data["Low"]

        futureReturn = data["Close"].pct_change(2).shift(-1)

        data["Outcome"] = (futureReturn > self.threshold).astype(int)

        relevantCols = [
            "Lagged_Return_1",
            "Lagged_Return_2",
            "Volatility",
            "Return_1",
            "Return_3",
            "Return_5",
            "Moving_Average",
            "Candle_Range"
        ]

        self.relevantCols = relevantCols

        self.data = data
        self.data_clean = data.dropna(subset=relevantCols + ["Outcome"]).copy()
        self.x = self.data_clean[relevantCols]
        self.y = self.data_clean["Outcome"]

    def trainAndSplit(self):
        if self.type not in ["RFC", "XGB"]:
            print("\nThe trainAndSplit() function can not be found for this type.")
            return

        trainTestMultiplier = 0.8

        splitID = int(len(self.x) * trainTestMultiplier)

        self.xTrain = self.x.iloc[:splitID]
        yTrain = self.y.iloc[:splitID]

        self.xTest = self.x.iloc[splitID:]
        self.yTest = self.y.iloc[splitID:]

        if(self.type == "RFC"):

            model = RandomForestClassifier(n_estimators=100)
            model.fit(self.xTrain, yTrain)
            self.model = model
            self.yPredicted = model.predict(self.xTest)

        elif(self.type == "XGB"):

            dTrain = xgb.DMatrix(self.xTrain, yTrain, enable_categorical=True)
            dTest = xgb.DMatrix(self.xTest, self.yTest, enable_categorical=True)

            params = {
                "objective": "binary:logistic",
                "device": "cuda",
                "lambda": 1,
                "alpha": 0.5,
                "max_depth": 3
            }
            n = 100

            model = xgb.train(params=params, dtrain=dTrain, num_boost_round=n)
            self.model = model
            dPredicted = model.predict(dTest)
            self.yPredicted = dPredicted

    def dataEvaluation(self):
        if(self.type not in ["RFC", "XGB"]):
            print("\nThe dataEvaluation() function can not be found for this type.")
            return

        if(self.type == "RFC"):
            y_proba = self.model.predict_proba(self.xTest)
            confidenceThreshold = 0.6
            yPredictedConfident = (y_proba[:, 1] > confidenceThreshold).astype(int)
        elif(self.type == "XGB"):
            confidenceThreshold = 0.6
            yPredictedConfident = (self.yPredicted > confidenceThreshold)

        accuracy = accuracy_score(self.yTest, yPredictedConfident)
        classification_rep = classification_report(self.yTest, yPredictedConfident)

        print(f"Accuracy: {accuracy: .2f}")
        print("\nClassification Report:\n", classification_rep)

    def featureEvaluation(self):
        if(self.type != "RFC"):
            print("\nThe featureEvaluation() function can not be found for this type.")
            return
        import pandas as pd
        import matplotlib.pyplot as plt

        importances = self.model.feature_importances_
        feature_importance = pd.Series(importances, index=self.xTrain.columns)
        feature_importance.sort_values().plot(kind="barh", title="Feature Importances")
        plt.tight_layout()
        plt.show()

        import seaborn as sns

        corr_matrix = self.x.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.show()

    def simulate(self):

        cerebro = bt.Cerebro()
        if(self.type in ["RFC", "XGB"]):
            confidenceThreshold = 0.6
            def get_signal(prob):
                if prob > 0.6:
                    return 1
                elif prob < 0.4:
                    return -1
                else:
                    return 0

            if(self.type == "RFC"):
                self.data_clean["signal_prob"] = self.model.predict_proba(self.x)[:, 1]
                #self.data_clean["signal"] = (self.data_clean["signal_prob"] > confidenceThreshold).astype(int)
                self.data_clean["signal"] = self.data_clean["signal_prob"].apply(get_signal)
            elif (self.type == "XGB"):
                dAll = xgb.DMatrix(self.data_clean[self.relevantCols], enable_categorical=True)
                self.data_clean["signal_prob"] = self.model.predict(dAll)
                #self.data_clean["signal"] = (self.data_clean["signal_prob"] > confidenceThreshold).astype(int)
                self.data_clean["signal"] = self.data_clean["signal_prob"].apply(get_signal)
            class PandasDataWithSignal(bt.feeds.PandasData):
                lines = ("signal",)
                params = (("signal", "signal"),)

            data_feed = PandasDataWithSignal(dataname=self.data_clean)

            class Strategy(bt.Strategy):
                params = (('signal_column', "signal"),)

                def __init__(self):
                    self.signal = getattr(self.datas[0].lines, self.p.signal_column)

                def next(self):
                    signal = self.signal[0]
                    if not self.position:
                        if signal == 1:
                            self.buy()
                        elif signal == -1:
                            self.sell()

                    elif self.position.size > 0:  # currently long
                        if signal <= 0:
                            self.close()
                        if signal == -1:
                            self.sell()

                    elif self.position.size < 0:  # currently short
                        if signal >= 0:
                            self.close()
                        if signal == 1:
                            self.buy()
            cerebro.addstrategy(Strategy, signal_column="signal")

        else:

            import os
            os.add_dll_directory(r"C:\msys64\ucrt64\bin")

            import bsm
            import numpy as np

            data_feed = bt.feeds.PandasData(dataname=self.data_clean)

            class Strategy(bt.Strategy):
                def __init__(self):
                    self.dataClose = self.datas[0].close
                    self.priceHistory = []
                    self.zScores = []

                def next(self):
                    current_price = self.dataClose[0]
                    self.priceHistory.append(current_price)
                    if (len(self.priceHistory) < 90):
                        return
                    prices = np.array(self.priceHistory[-90:])
                    logReturns = np.diff(np.log(prices))
                    RFI = 0.05
                    maturity = 30 / 252

                    #Parameters
                    tradingDays = 252

                    ap = current_price
                    sp = current_price
                    vol = np.std(logReturns) * np.sqrt(tradingDays)
                    threshold = 3 * np.std(logReturns)
                    jumps = logReturns[np.abs(logReturns) > threshold]
                    jumpIntensity = len(jumps) / len(logReturns) * tradingDays if len(jumps) > 0 else 0 #252
                    mu_j = np.mean(jumps) if len(jumps) > 0 else 0
                    delta = np.std(jumps) if len(jumps) > 0 else 0
                    m = np.exp(mu_j + 0.5 * delta ** 2) if jumpIntensity > 0 else 1.0
                    dividEnds = 0.0

                    myModel = bsm.Merton(ap, sp, maturity, vol, RFI, jumpIntensity, m, delta, dividEnds)
                    calculatedPrices = myModel.computeWithThreads(10000)

                    deviation = (calculatedPrices.CallPrice - calculatedPrices.PutPrice) - np.mean(prices)
                    standardDeviation = np.std(prices)
                    zScore = deviation / standardDeviation
                    self.zScores.append(zScore)

                    if len(self.zScores) > 20:
                        zsArray = np.array(self.zScores[-20:])
                        upperThreshold = np.percentile(zsArray, 90)
                        lowerThreshold = np.percentile(zsArray, 10)
                    else:
                        return
                    if not self.position:
                        assetScale = 1.0
                        if zScore >= upperThreshold:
                            size = int(self.broker.getcash() * assetScale / current_price)
                            if size > 0:
                                self.buy(size=size)
                        elif zScore <= lowerThreshold:
                            size = int(self.broker.getcash() * assetScale / current_price)
                            if size > 0:
                                self.sell(size=size)
                    elif self.position:
                        zs_array = np.array(self.zScores[-20:])
                        exitUpper = np.percentile(zsArray, 70)
                        exitLower = np.percentile(zsArray, 30)
                        if exitLower < zScore < exitUpper:
                            self.close()

            cerebro.addstrategy(Strategy)

        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.00)

        startCash = cerebro.broker.getvalue()
        print(f"\nStarting Portfolio Value: ${startCash: .2f}")
        cerebro.run()
        finalCash = cerebro.broker.getvalue()
        print(f"\nFinal Portfolio Value: ${finalCash: .2f}")

        difference = finalCash - startCash

        if (difference >= 0):
            print(f"\nProfit: ${difference: .2f}")
        else:
            print(f"\nLost: ${difference: .2f}")
        if(startCash != 0):
            print(f"\nReturn: {(finalCash / startCash - 1) * 100: .2f}%")





