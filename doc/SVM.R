
library(gbm)
library(caret)
library(rpart)
library(e1071)
setwd("~/GitHub/Fall2016-proj3-grp1/data/Project3_poodleKFC_train")



feature_sift<-read.csv("sift_features.csv")
feature_sift_new<-t(feature_sift)
sum(is.na(feature_sift_new))
y <- list.files("images/")
y<- gsub('.{9}$', '', y)
y <- as.factor(y)

data<-as.data.frame(cbind(feature_sift_new,y))


index <- 1:nrow(data)
testindex <- sample(index, trunc(length(index)/3))
testset <- data[testindex,]
trainset <-data[-testindex,]


system.time(svm.model <- svm(y ~ ., data = trainset, cost = 150, gamma = 1, scale = FALSE, type = "C-classification"))
svm.pred <- predict(svm.model, testset[,-5001])
pred <-as.matrix(svm.pred)
confusionMatrix(pred, testset$y)

