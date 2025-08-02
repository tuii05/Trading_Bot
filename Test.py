import Trading_Bot

mybot = Trading_Bot.tradingBot("META", "Merton")
mybot.createData()
mybot.trainAndSplit()
mybot.dataEvaluation()
mybot.featureEvaluation()
mybot.simulate()