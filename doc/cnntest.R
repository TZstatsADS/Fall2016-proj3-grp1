setwd("G:/Columbia/STAT GR5243/project03")
library(e1071)
load("CNN.RData")
train<-dat[1:1600,]
test<-dat[1601:2000,]
cv.error.rate<-vector()
cost.set<-c(0.001,0.005,0.01,0.05,0.1,0.5,1)
for(i in cost.set){
  soft.svm.cv<-svm(V1~.,data=train,type='C-classification',kernel='linear',cost=i,cross=5)
  cv.error.rate<-cbind(cv.error.rate,1-(soft.svm.cv$tot.accuracy/100))
}
soft.svm<-svm(V1~.,data=train,type='C-classification',kernel='linear',cost=cost.set[which.min(cv.error.rate)])
predict.svm<-predict(soft.svm,test[,-1])
print(table(pred=predict.svm,true=test[,1]))

cost.set.new<-c(1,5,10,20,50)
cv.error.rate.c<-vector()
gamma.set<-c(0.000005,0.00001,0.00005,0.0001,0.0005)
for(i in cost.set.new){
  cv.error.rate.g<-vector()
  for(j in gamma.set){
    kernel.svm.cv<-svm(V1~.,data=train,type='C-classification',kernel='radial',cost=i,gamma=j,cross=5)
    cv.error.rate.g<-rbind(cv.error.rate.g,1-(kernel.svm.cv$tot.accuracy/100))
  }
  cv.error.rate.c<-cbind(cv.error.rate.c,cv.error.rate.g)
}
kernel.svm<-svm(V1~.,data=train,type='C-classification',kernel='radial',cost=cost.set.new[which(cv.error.rate.c==min(cv.error.rate.c),arr.ind=TRUE)[1,2]],gamma=gamma.set[which(cv.error.rate.c==min(cv.error.rate.c),arr.ind=TRUE)[1,1]])
predict.svm.kernel<-predict(kernel.svm,test[,-1])
print(table(pred=predict.svm.kernel,true=test[,1]))
