library(e1071)
library(gbm)
# please change the path
setwd("G:/Columbia/STAT GR5243/project03")
# please change the training data set
# note training data should be a data frame with a variable y as lable variable
load("SURF+Color.RData")
y<-rep(c(1,0),each=1000)
xgboost_data=t(t(surf))
surf<-surf<-cbind(surf,y)
# function for model training: 
# format for train_data should be data frame
# columns for train_data should be variables and rows for train_data should be data points
# there should be a column named y contains the labels of all data points
# method_all should be one of the followings: gbm, svm and xgboost
# method_svm should be one of the followings: linear, radial
model_train<-function(train_data,method_all,method_svm){
  if(method_all=="gbm"){
    gbm_base<-list()
    gbm_baseline<-function(train,dep){
      gbm_model<-gbm(y~.,data=train,shrinkage=0.01,distribution="bernoulli",cv.folds=5,n.trees=2000,interaction.depth=dep,verbose=F)
      return(gbm_model)
    }
    for(depth in c(1,3,5)){
      gbm_base[[(depth+1)/2]]<-gbm_baseline(train_data,depth)
    }
    return(gbm_base)
  } else if(method_all=="svm"){
    if(method_svm=="linear"){
      cv.error.rate<-vector()
      cost.set<-c(0.1,0.5,1,5,10,50)
      for(i in cost.set){
        soft.svm.cv<-svm(y~.,data=train_data,type='C-classification',kernel='linear',cost=i,cross=5)
        cv.error.rate<-cbind(cv.error.rate,1-(soft.svm.cv$tot.accuracy/100))
      }
      soft.svm<-svm(y~.,data=train_data,type='C-classification',kernel='linear',cost=cost.set[which.min(cv.error.rate)])
      return(soft.svm)
    } else if(method_svm=="radial"){
      cost.set.new<-c(1,5,10,20,50)
      cv.error.rate.c<-vector()
      gamma.set<-c(0.01,0.1,1,5,10)
      for(i in cost.set.new){
        cv.error.rate.g<-vector()
        for(j in gamma.set){
          kernel.svm.cv<-svm(y~.,data=train_data,type='C-classification',kernel='radial',cost=i,gamma=j,cross=5)
          cv.error.rate.g<-rbind(cv.error.rate.g,1-(kernel.svm.cv$tot.accuracy/100))
        }
        cv.error.rate.c<-cbind(cv.error.rate.c,cv.error.rate.g)
      }
      kernel.svm<-svm(y~.,data=train_data,type='C-classification',kernel='radial',cost=cost.set.new[which(cv.error.rate.c==min(cv.error.rate.c),arr.ind=TRUE)[1,2]],gamma=gamma.set[which(cv.error.rate.c==min(cv.error.rate.c),arr.ind=TRUE)[1,1]])
      return(kernel.svm)
    } else{
      print("no such a model exists")
    }
    
  } else if(method_all=="xgboost"){
    # set up the cross-validated hyper-parameter search
    xgb_grid_1=expand.grid(
      nrounds=100,
      eta=c(0.01, 0.001, 0.0001),
      max_depth=c(2, 4,6,8),
      gamma=1,
      colsample_bytree=0.6,
      min_child_weight=0.5
    )
    # pack the training control parameters
    xgb_trcontrol_1=trainControl(
      method="cv",
      number=5,  
      allowParallel=TRUE
    )
    # train the model for each parameter combination in the grid, 
    # using CV to evaluate
    xgb_train_1=train(
      x=xgboost_data,
      y=as.factor(y),
      trControl=xgb_trcontrol_1,
      tuneGrid=xgb_grid_1,
      method="xgbTree"
    )
    para=xgb_train_1$bestTune
    xgb_train=xgboost(data=xgboost_data,label=y,nrounds=para$nrounds,eta=para$eta,max_depth=para$max.depth,gamma=para$gamma)
    return(xgb_train)
  } else{
    print("no such a model exists")
  }
}
# baseline model training
ptm_gbm<-proc.time()
train_return_base<-model_train(surf,"gbm","linear")
proc.time()-ptm_gbm
# svm model training
ptm_svm<-proc.time()
train_return<-model_train(surf,"svm","linear")
proc.time()-ptm_svm