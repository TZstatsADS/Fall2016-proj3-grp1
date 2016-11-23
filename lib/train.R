train <- function(dat_train, label_train){
  library(gbm)
  gbm_model<-gbm(label_train~.,data=dat_train,shrinkage=0.01,distribution="bernoulli",cv.folds=5,n.trees=2000,interaction.depth=5,verbose=F)
  gbm_iter<-gbm.perf(gbm_model,method="cv")
  
  library(e1071)
  cv_error_rate<-vector()
  cost_set<-c(0.01,0.1,0.5,1,5,10,20,50,100)
  for(i in cost_set){
    soft_svm_cv<-svm(label_train~.,data=dat_train,type='C-classification',kernel='linear',cost=i,cross=5)
    cv_error_rate<-cbind(cv_error_rate,1-(soft_svm_cv$tot.accuracy/100))
  }
  soft_svm<-svm(label_train~.,data=dat_train,type='C-classification',kernel='linear',cost=cost_set[which.min(cv_error_rate)])
  
  return(list(baseline=gbm_model,baseline_iter=gbm_iter,adv=soft_svm))
}