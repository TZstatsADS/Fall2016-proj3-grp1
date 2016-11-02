######XGBOOST
#train model
library(Matrix)
require(xgboost)

#read feature into sparse matrix
feature_sift<-read.csv("sift_features.csv")
feature_sift_new<-t(feature_sift)
feature_sift_new = Matrix(feature_sift_new, sparse = T) 

y <- list.files("images/")
y<- gsub('.{9}$', '', y)
y<-as.numeric(unlist(y) == "dog")

#train test split 
num_image = NROW(feature_sift_new)
train_index = sample(seq(1:num_image), 0.7 * num_image)
train_x = feature_sift_new[train_index,]
train_y = y[train_index]
test_x = feature_sift_new[-train_index, ]
test_y = y[-train_index]

#tune model
bst = xgboost(data = train_x,  label = train_y, 
              max.depth = 4, eta = 1, nthread = 2, nround = 20,
              objective = "binary:logistic")


pred  = predict(bst, test_x)
#pred is in the form of probability, transform into class
prediction = as.numeric(pred > 0.5)
sum(prediction == test_y)/length(test_y)

#####tune
library(caret)
xgb_params_1 = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 3,                                                               # max tree depth
  eval_metric = "auc"                                                          # evaluation/loss metric
)
# fit the model with the arbitrary parameters specified above
xgb_1 = xgboost(data = train_x,
                label = train_y,
                params = xgb_params_1,
                nrounds = 100,                                                 # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 1,
                early.stop.round = 10                                          # stop if no improvement within 10 trees
)
# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = train_x,
                  label = train_y,
                  nrounds = 100, 
                  nfold = 5,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  showsd = TRUE,                                               # standard deviation of loss across folds
                  stratified = TRUE,                                           # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1, 
                  early.stop.round = 10
)


# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1,
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight=c(0.5, 1, 1.5)
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
 time_start <-Sys.time()
xgb_train_1 = train(
  x = train_x,
  y = as.factor(train_y),
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)
time_end <- Sys.time()
#####What I get for best parameter
#train model
bst = xgboost(data = train_x,  label = train_y,max.depth = 4, eta = 0.5, nthread = 2, nround = 47,objective = "binary:logistic")

pred  = predict(bst, test_x)
#pred is in the form of probability, transform into class
prediction = as.numeric(pred > 0.5)
sum(prediction == test_y)/length(test_y)
