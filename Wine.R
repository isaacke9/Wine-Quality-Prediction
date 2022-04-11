#STAT 656 Project

library(neuralnet)
library(nnet)
library(NeuralNetTools)
library(clusterGeneration)
library(e1071)
library(devtools)
library(caret)
library(rminer)
source_gist('6206737')

wine = read.csv("WineQT.csv")
wine = wine[,c(1:12)]
wine[,12] = as.factor(wine[,12])

set.seed(1)
train = sample(1:nrow(wine),nrow(wine)*3/4,replace = F)
wine.train = wine[train,]
wine.test = wine[-train,]

################################################################################
#Neural Network
labels = c("3", "4", "5", "6", "7", "8") 

#Train/Test
nn.mod = neuralnet(quality~., data = wine.train, linear.output = F)
nn.1 = compute(nn.mod, wine.test)
nn.2 = max.col(nn.1$net.result)
nn.3 = labels[nn.2]
nn.4 = ifelse(nn.3 == wine.test$quality, 0, 1)
nn.test.error = sum(nn.4)/nrow(wine.test)

#K-Fold CV
K = 10
s = nrow(wine)/K
j = 1
error = NULL

for(i in 1:K){
  valid = as.data.frame(wine[j:(i*s),])
  train = wine[-c(as.numeric(rownames(valid))),]
  nn.data = neuralnet(quality~., data = train, linear.output = F)
  nn.pred = compute(nn.data,valid)
  nn.pred2 = max.col(nn.pred$net.result)
  nn.pred3 = labels[nn.pred2]
  nn.pred4 = ifelse(nn.pred3 == valid$quality, 0, 1)
  error[i] = sum(nn.pred4)/nrow(valid)
  j = i*s + 1
}

nn.cv.error = sum(error)/K

#Feature Importance
nnet.mod = nnet(quality~., data = wine.train, size = 11, linout = F)
nnet.pred = predict(nnet.mod, wine.test, type = "class")
nnet.table = table(nnet.pred,wine.test$quality)

gar.fun('quality',nnet.mod)

plot(nn.mod)
################################################################################

################################################################################
#SVM

#Train/Test
svm.fit = svm(quality~., data = wine.train, type = "C-classification", kernel = "linear", cost = .1)
svm.1 = predict(svm.fit, wine.test)
svm.2 = ifelse(svm.1 == wine.test$quality, 0, 1)
svm.test.error = sum(svm.2)/nrow(wine.test)

#K-Fold CV
K = 10
s = nrow(wine)/K
j = 1
error = NULL

for(i in 1:K){
  valid = as.data.frame(wine[j:(i*s),])
  train = wine[-c(as.numeric(rownames(valid))),]
  svmfit = svm(quality~.,data=train, type = "C-classification", kernel= 'linear', cost=.1)
  svm.pred = predict(svmfit, valid)
  svm.pred2 = ifelse(svm.pred == valid$quality, 0, 1)
  error[i] = sum(svm.pred2)/nrow(valid)
  j = i*s + 1
}

svm.error = sum(error)/K

#Feature Importance
w = t(svm.fit$coefs) %*% svm.fit$SV
w = apply(w, 2, function(v){sqrt(sum(v^2))})
w = sort(w, decreasing = T)
w
################################################################################







