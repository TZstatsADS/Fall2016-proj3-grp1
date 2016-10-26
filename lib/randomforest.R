library(sgd)
library(MASS)

sift <- read.csv(file="sift_features.csv",head = T, sep = ",")
sift_features <- matrix(unlist(sift), byrow= T, nrow= 2000)
y = rep.int(0,2000)
y[1:1000] = 1   # 1= chikcen 0 = dog
dat <- data.frame(y=y,x = sift_features)

splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)*0.8))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}

splits <- splitdf(dat, seed = 808)
lapply(splits, nrow)
training <- splits$trainset
testing <- splits$testset

#random forest
library(randomForest)
model <- randomForest(y~., 
                      data = training, 
                      importance=TRUE,
                      keep.forest=TRUE
)
print(model)
varImpPlot(model, type=1)
predicted <- predict(model, newdata=testing[ ,-1])
predicted <- round(predicted)
mean(predicted == testing$y)



