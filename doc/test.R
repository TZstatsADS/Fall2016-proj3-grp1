# train_object is the model returned from train.R
# test_data should not contain the label column
model_test<-function(test_data,train_object){
  predict<-predict(train_object,test_data)
  return(predict)
}
test_return<-model_test(surf[,-ncol(surf)],train_return)
# checking accuracy
print(table(pred=test_return,true=surf[,ncol(surf)]))