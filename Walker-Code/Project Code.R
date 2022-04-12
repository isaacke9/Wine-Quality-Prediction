library(caret)
library(dplyr)
library(ggplot2)
library(earth)
library(vip)
library(glmnet)
library(randomForest)

wine = read.csv("WineQT.csv")

# train/test split
set.seed(2021)
trainIndex = createDataPartition(wine$quality,p=.75,list=FALSE)
Xtrain = wine[trainIndex,-c(12,13)]
Ytrain = make.names(wine[trainIndex,12])
Xtest = wine[-trainIndex,-c(12,13)]
Ytest = make.names(wine[-trainIndex,12])

# fda
fda = train(x = Xtrain, 
            y = Ytrain,
            method = 'fda',
            metric = 'Accuracy',
            tuneGrid = expand.grid(degree = 1:3, nprune = c(5,10,20,50,100)),
            trControl = trainControl(method='CV',number = 5, classProbs = TRUE))
plot(fda)

fdaVip = vip(fda,num_features = 11, bar = FALSE, metric = "Accuracy")
plot(fdaVip)

fda.pred = predict(fda$finalModel, Xtest)
table(fda.pred,Ytest)
confusionMatrix(factor(fda.pred),factor(Ytest)) # 57.75%

# mars
mars = train(x = Xtrain,
             y = Ytrain,
             method = 'earth',
             metric = 'Accuracy',
             tuneGrid = expand.grid(degree = 1:3, nprune = c(5,10,20,50,100)),
             trControl = trainControl(method='CV',number = 5, classProbs = TRUE))
plot(mars)

marsVip = vip(mars,num_features = 11, bar = FALSE, metric = "Accuracy")
plot(marsVip)

mars.pred = predict(mars$finalModel, Xtest, type="class")
table(mars.pred,Ytest)
confusionMatrix(factor(mars.pred),factor(Ytest)) # 55.99%

# random forest
rf = randomForest(quality~.,wine[trainIndex,-13])

varImpPlot(rf)

rf.pred = make.names(round(predict(rf,Xtest)))
table(rf.pred,Ytest)
confusionMatrix(factor(rf.pred),factor(Ytest)) # 62.32%

rfTune = tuneRF(x = Xtrain,
                y = factor(wine[trainIndex,12]),
                ntreeTry = 500,
                mtryStart = 2,
                stepFactor = 1.5,
                improve = .01,
                trace = TRUE)
plot(rfTune)