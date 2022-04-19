#STAT 654 Project

library(randomForest)
library(pROC)
library(partykit)]
library(party)

wine = read.csv("WineQT.csv")
wine = wine[,c(1:12)]
wine$quality = ifelse(wine$quality <= 5, 0, 1)

set.seed(1)
train = sample(1:nrow(wine),nrow(wine)*0.8,replace = F)
wine.train = wine[train,]
wine.test = wine[-train,]

################################################################################
#Random Forest
#Test/Train
set.seed(101)
error = NULL
for(i in 1:20){
  rf.data = randomForest(as.factor(quality)~.,data=wine.train, type = "classification", ntree = i*50)
  rf.pred = as.numeric(predict(rf.data,wine.test,type="class"))
  rf.pred = ifelse(as.numeric(rf.pred) == 1, 0, 1)
  error[i] = sum(abs(rf.pred - wine.test$quality))/nrow(wine.test)
}
which.min(error[i])

#ROC Curve
rf_roc = roc(rf.pred, wine.test$quality)
auc(rf_roc)
confusionMatrix(as.factor(rf.pred), as.factor(wine.test$quality))

varImpPlot(rf.data)
cforest = cforest(quality ~ ., data=wine, controls=cforest_control(mtry=2, mincriterion=0))
pt <- prettytree(cf@ensemble[[1]], names(cf@data@get("input"))) 
nt <- new("BinaryTree") 
nt@tree <- pt 
nt@data <- cf@data 
nt@responses <- cf@responses 

plot(nt, type="simple")
plot(rf.data, log="y")

#CV
K = 5
s = nrow(wine)/K
j = 1
error = NULL
for(i in 1:K){
  valid = as.data.frame(wine[j:(i*s),])
  train = wine[-c(as.numeric(rownames(valid))),]
  rf.data = randomForest(as.factor(quality)~.,data=train, type = "classification")
  rf.pred = as.numeric(predict(rf.data,valid,type="class"))
  rf.pred = ifelse(as.numeric(rf.pred) == 1, 0, 1)
  error[i] = sum(abs(rf.pred - valid$quality))/nrow(valid)
  j = i*s + 1
}
sum(error)/K

#CV Caret
trainControl = trainControl(method = "cv", number = 5)
rfOut = train(x = wine.train[,-12], y = wine.train[,12], method = 'rf', metric = 'Accuracy', trControl = trainControl)
plot(lassoOut, xlab = "Penalty", ylab = 'K-fold CV')

#Visualization
getTree(randomForest(wine[,-12], wine[,12], ntree=500))
################################################################################
