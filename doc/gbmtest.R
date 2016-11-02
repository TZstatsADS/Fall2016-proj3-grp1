ptm<-proc.time()
library(gbm)
library(caret)
# 改一下路径 change the work directory
setwd("G:/Columbia/STAT GR5243/project03")
# 改一下data 改成sift的data change the data set
load("SURF.RData")
# data格式：前1000行是chicken 后1000行是dog data format: first 1000 lines are chicken, last 1000 lines are dogs
y<-rep(c(1,0),each=1000)
# 改一下名称 change the name
surf<-cbind(surf,y)
train.surf<-surf[c(1:800,1001:1800),]
test.surf<-surf[c(801:1000,1801:2000),]
# 以下应该不用改 do not need to change
info<-vector()
gbm_baseline<-function(train,test,dep){
  gbm_model<-gbm(y~.,data=train,shrinkage=0.01,distribution="bernoulli",cv.folds=5,n.trees=2000,interaction.depth=dep,verbose=F)
  best_iter<-gbm.perf(gbm_model,method="cv")
  test_acc<-confusionMatrix(predict(gbm_model,test)>0,test$y>0)
  train_acc<-confusionMatrix(predict(gbm_model,train)>0,train$y>0)
  return(c(best_iter,train_acc$overall[1],test_acc$overall[1]))
}
for(depth in c(1, 3, 5)){
  gbm_base<-gbm_baseline(train.surf,test.surf,depth)
  info<-rbind(info,gbm_base)
}
proc.time()-ptm