
library(gbm)
library(caret)
library(rpart)
library(e1071)
library(kernlab)    
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


system.time(svm_tune <- tune(svm, train.x=trainset[,-5001], train.y=trainset$y, scale = FALSE,
     kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2))))

ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)

svm.tune <-train(x=trainset[,-5001],
      y= trainset$y,
      method = "svmRadial",   # Radial kernel
      tuneLength = 9,					# 9 values of the cost function
      preProc = c("center","scale"),  # Center and scale data
      metric="ROC",
      trControl=ctrl)
system.time(svm1 <- ksvm(y~.,data=trainset,kernel="polydot",kpar="automatic",C=60,cross=3,prob.model=FALSE, scaled = FALSE, type = "C-svc"))
svm1
system.time(kernlab_p <- predict(svm1, testset[,-5001]))
confusionMatrix(kernlab_p, testset$y)
