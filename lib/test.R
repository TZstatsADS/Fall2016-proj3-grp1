test <- function(mod_train, dat_test){
  library(gbm)
  baseline<-predict(mod_train$baseline,dat_test,mod_train$baseline_iter)
  baseline[baseline>0]<-1
  baseline[baseline<=0]<-0
  library(e1071)
  adv<-predict(mod_train$adv,dat_test)
  adv<-as.numeric(adv)
  adv[adv==1]<-0
  adv[adv==2]<-1
  return(list(baseline=baseline,adv=adv))
}