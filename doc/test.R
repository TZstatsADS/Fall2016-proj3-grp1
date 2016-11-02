load("Test_dat.RData")
# train_object is the model returned from train.R
# test_data should not contain the label column
model_test<-function(test_data,train_object){
  predict<-predict(train_object,test_data)
  return(predict)
}
# gbm prediction
test_return_base<-model_test(dat,train_return_base[[3]])
# svm prediction
test_return<-model_test(dat,train_return)
# checking accuracy
# please change the parameter true when you test the accuracy
# gbm accuracy test
test_return_base[test_return_base>0]<-1
test_return_base[test_return_base<0]<-0
print(table(pred=test_return_base,true=surf[,ncol(surf)]))
# svm accuracy test
print(table(pred=test_return,true=surf[,ncol(surf)]))
# write predicted labels as .csv files
write.csv(test_return_base,file="gbm.csv")
write.csv(test_return,file="svm.csv")
