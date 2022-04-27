library(caret)
library(dplyr)
library(glmnet)
library(pROC)

wine = read.csv("WineQT.csv") # load in data
wine$quality = factor(wine$quality) # convert response to factor

# train/test split
set.seed(2021)
trainIndex = createDataPartition(wine$quality,p=.8,list=FALSE)
Xtrain = wine[trainIndex,-c(12,13)]
Ytrain = wine[trainIndex,12]
Xtest = wine[-trainIndex,-c(12,13)]
Ytest = wine[-trainIndex,12]

# tune probit parameters
trainControl = trainControl(method="cv",number=5)
tuneGrid = expand.grid('alpha'=c(0,.25,.5,.75,1),'lambda'=seq(.0001,.01,length.out=10))
probit = train(x = Xtrain,
             y = Ytrain,
             method = "glmnet",
             family = "binomial",
             link = "probit",
             metric = "Accuracy",
             trControl = trainControl,
             tuneGrid = tuneGrid)
plot(probit)
max(probit$results$Accuracy)
probit$results[which.max(probit$results$Accuracy),]

# fit model
probit.fit = glmnet(x = as.matrix(Xtrain),
                    y = Ytrain,
                    alpha = probit$bestTune$alpha,
                    family = binomial(link="probit"))
# predict on test set
probit.prob = predict(probit.fit,
                      as.matrix(Xtest),
                      s = probit$bestTune$lambda,
                      type = 'response')
probit.pred = ifelse(probit.prob > 0.5, 1, 0)

confusionMatrix(as.factor(probit.pred),Ytest) # 71.05%

# roc curve
probit.pred.roc = predict(probit,
                          Xtest,
                          s = probit$bestTune$lambda,
                          type = 'prob')
probit.roc = roc(response = Ytest,probit.pred.roc$`1`)
plot(probit.roc)
auc(probit.roc)

##########

# test 100 different combinations
alpha = NULL
lambda = NULL
accuracy = NULL

for(i in 1:100){
  print(i)
  probit = train(x = Xtrain,
                 y = Ytrain,
                 method = "glmnet",
                 family = "binomial",
                 link = "probit",
                 metric = "Accuracy",
                 trControl = trainControl,
                 tuneGrid = tuneGrid)
  alpha[i] = probit$bestTune$alpha
  lambda[i] = probit$bestTune$lambda
  
  probit.fit = glmnet(x = as.matrix(Xtrain),
                      y = Ytrain,
                      alpha = probit$bestTune$alpha,
                      family = binomial(link="probit"))
  probit.prob = predict(probit.fit,
                        as.matrix(Xtest),
                        s = probit$bestTune$lambda,
                        type = 'response')
  probit.pred = ifelse(probit.prob > 0.5, 1, 0)
  
  accuracy[i] = confusionMatrix(as.factor(probit.pred),Ytest)$overall[1]
}

table(alpha)
table(lambda)
hist(accuracy)
cbind(alpha,lambda,accuracy)

mean(accuracy)
quantile(accuracy,c(.025,.975))

##########

# variable importance
barplot(coef(probit.fit,s=probit$bestTune$lambda)[-1][as.vector(abs(coef(probit.fit,s=probit$bestTune$lambda)[-1])>1e-16)],
        horiz=T,cex.names=.5,las=1,names.arg=names(wine)[c(1:7,9:11)],xlab='Coefficient')