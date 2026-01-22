setwd("/Users/marcelpotinga/Downloads/Social-Media_fake_Profiles")
library(readxl)
data = read_excel("fake_social_media_global_2.0_with_missing.xlsx")
data<- as.data.frame(data)
##PRE PROCESSING

#Counting Missing values
missing_values = colSums(is.na(data))/nrow(data)
print(missing_values)
#If we remove all the data, we are left with very little data so we will use RFR as decided in the report
#We have categorical that is transformed to NUM, so we have to factor it
colonne <- c("has_profile_pic", "username_randomness","suspicious_links_in_bio","verified")  # le colonne che scegli tu

dati_dummy <- data[colonne]
dati_dummy[] <- lapply(dati_dummy, factor, levels = c(0,1))
data[colonne] <- dati_dummy

#We will now drop platform and username as they are not useful for the model
data$platform <- NULL
data$username <- NULL

#RFR Imputation
library(caret)
data_to_impute <- data
data_to_impute$is_fake <- NULL #remove target variable for imputation

library(missForest)

set.seed(196)
data_to_impute.rf <- missForest(data_to_impute)

#check imputed values
head(data_to_impute.rf$ximp)
# do all also factor vars

# save data imputed
imputedRF=data.frame(data_to_impute.rf$ximp)
imputedRF$is_fake=data$is_fake

#check missing values after imputation
missing_values_after_imputation = colSums(is.na(imputedRF))/nrow(imputedRF)
print(missing_values_after_imputation)


#Missing data OK!
#Next check NZV!
library(caret)
nzv = nearZeroVar(imputedRF, saveMetrics = TRUE)
head(nzv[order(nzv$percentUnique, decreasing = FALSE), ], n = 20)
#No NZV found!

#Next Collinearity, we need Quantitative_df
Quantitative_df <- imputedRF[, sapply(imputedRF, is.numeric)]

correlatedPredictors = findCorrelation(cor(Quantitative_df), cutoff = 0.95, names=TRUE)
correlatedPredictors
#Let's plot the correlation matrix to visualize better
library(corrplot)
corrplot(cor(Quantitative_df), method = "circle", type = "upper", tl.cex = 0.7)

#Problem is digit_ratio with digit count so we will directly drop digit_count
imputedRF$digits_count <- NULL

#Ok!

library("detectseparation")
endo_sep <- glm(is_fake ~ ., data = imputedRF,
                family = binomial("logit"),
                method = "detect_separation")
endo_sep
#No separation issues found!
#Final check on the target variable distribution
table(imputedRF$is_fake)
#Balanced dataset!
#All pre-processing steps done!


train_idx <- createDataPartition(imputedRF$is_fake, p = 0.8, list = FALSE)
train <- imputedRF[train_idx, ]
test  <- imputedRF[-train_idx, ]
train$is_fake <- factor(train$is_fake, 
                         levels = c("0", "1"), 
                         labels = c("zero", "one"))
test$is_fake <- factor(test$is_fake, 
                         levels = c("0", "1"), 
                         labels = c("zero", "one"))
table(train$is_fake)
table(test$is_fake)
#MS DT + no Scaing
set.seed(1)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE)
rpartTuneCvA <- train(is_fake ~ ., data = train, method = "rpart",
                      tuneLength = 10,
                      trControl = cvCtrl)


# select only important variables
vi=as.data.frame(rpartTuneCvA$finalModel$variable.importance)
vi

#Create dataset with only these features
selected_features <- rownames(vi)
train_dt <- train[, c(selected_features, "is_fake")]
test_dt <- test[, c(selected_features, "is_fake")]

#LASSO 
set.seed(1)
cvCtrl_lasso <- trainControl(method = "cv", number = 5,search="grid", classProbs = TRUE)
lasso_puro <- train(
  is_fake ~ ., 
  data = train,
  method = "glmnet",
  family = "binomial",
  trControl = cvCtrl_lasso,
  tuneGrid = expand.grid(  # alpha=1 (Lasso), 10 valori lambda
    alpha = 1,
    lambda = seq(0.001, 0.1, length = 10)
  ),
  preProcess = c("center", "scale")
)


keep_vars <- c("digit_ratio", "username_length", "special_char_count", "repeat_char_count")
train_lasso <- train[, c(keep_vars, "is_fake")]
test_lasso <- test[, c(keep_vars, "is_fake")]
#Lasso+STD
pre_lasso_std <-preProcess(train_lasso, method = c("center", "scale"))
train_lasso_std <- predict(pre_lasso_std, train_lasso)
test_lasso_std  <- predict(pre_lasso_std, test_lasso)

#DT + STD
pre_dt_std <-preProcess(train_dt, method = c("center", "scale"))
train_dt_std <- predict(pre_dt_std, train_dt)
test_dt_std  <- predict(pre_dt_std, test_dt)

#Lasso+STD+bc

pre_lasso_std_bc <-preProcess(train_lasso, method = c("center", "scale","BoxCox"))
train_lasso_std_bc <- predict(pre_lasso_std_bc, train_lasso)
test_lasso_std_bc  <- predict(pre_lasso_std_bc, test_lasso)

#DT + STD + bc
pre_dt_std_bc <-preProcess(train_dt, method = c("center", "scale","BoxCox"))
train_dt_std_bc <- predict(pre_dt_std_bc, train_dt)
test_dt_std_bc  <- predict(pre_dt_std_bc, test_dt)

#Train/Test only STD
train_only_std <- preProcess(train, method= c("center", "scale"))
train_std <- predict(train_only_std, train)
test_std <- predict(train_only_std, test)

#Train/Test only STD + bc
train_only_std_bc <- preProcess(train, method= c("center", "scale","BoxCox"))
train_std_bc <- predict(train_only_std_bc, train)
test_std_bc <- predict(train_only_std_bc, test)


#Modelling on DT MS no Scaling

#metric <- "ROC"
#control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE,savePrediction = TRUE)
# random_forest <- train(is_fake~., data=train_dt, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree <- train(is_fake ~ ., data = train_dt, method = "rpart",metric=metric,
#        tuneLength = 10,
#        trControl = control)
# xgboost_model <- train(is_fake ~ ., data = train_dt, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# logistic_regression <- train(is_fake ~ ., data = train_dt, method = "glm",family = "binomial",trControl = control,metric=metric,preProcess = c("center", "scale"))
# library(nnet)
# #ann_model <- train(is_fake ~ ., data = train_dt, method = "nnet",trControl = control,metric="ROC",preProcess = c("center", "scale"), trace = FALSE, linout = FALSE, maxit = 200)
# save(random_forest,decision_tree,xgboost_model,logistic_regression,ann_model, file="models_DT_NoPreProc.Rdata")
load("models_DT_NoPreProc.Rdata")
#Modelling on Lasso MS no SCALING
# random_forest_lasso <- train(is_fake~., data=train_lasso, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_lasso <- train(is_fake ~ ., data = train_lasso, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                        trControl = control)
# xgboost_model_lasso <- train(is_fake ~ ., data = train_lasso, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# logistic_regression_lasso <- train(is_fake ~ ., data = train_lasso, method = "glm",family = "binomial",trControl = control,metric=metric,preProcess = c("center", "scale"))
# ann_model_lasso <- train(is_fake ~ ., data = train_lasso, method = "nnet",trControl = control,metric="ROC",preProcess = c("center", "scale"), trace = FALSE, linout = FALSE, maxit = 200)
# save(random_forest_lasso,decision_tree_lasso,xgboost_model_lasso,logistic_regression_lasso,ann_model_lasso, file="models_Lasso_NoPreProc.Rdata")
load("models_Lasso_NoPreProc.Rdata")

#Modelling on DT + STD
# random_forest_dt_std <- train(is_fake~., data=train_dt_std, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_dt_std <- train(is_fake ~ ., data = train_dt_std, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                         trControl = control)
# xgboost_model_dt_std <- train(is_fake ~ ., data = train_dt_std, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# save(random_forest_dt_std,decision_tree_dt_std,xgboost_model_dt_std, file="models_DT_STD.Rdata")
load("models_DT_STD.Rdata")

#Modelling on Lasso + STD
# random_forest_lasso_std <- train(is_fake~., data=train_lasso_std, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_lasso_std <- train(is_fake ~ ., data = train_lasso_std, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                         trControl = control)
# xgboost_model_lasso_std <- train(is_fake ~ ., data = train_lasso_std, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# save(random_forest_lasso_std,decision_tree_lasso_std,xgboost_model_lasso_std, file="models_Lasso_STD.Rdata")
load("models_Lasso_STD.Rdata")

#Modelling on DT + STD + bc
# random_forest_dt_std_bc <- train(is_fake~., data=train_dt_std_bc, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                         trControl = control)
# xgboost_model_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
#logistic_regression_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "glm",family = "binomial",trControl = control,metric=metric)
# ann_model_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "nnet",trControl = control,metric="ROC", trace = FALSE, linout = FALSE, maxit = 200)
#save(random_forest_dt_std_bc,decision_tree_dt_std_bc,xgboost_model_dt_std_bc, ann_model_dt_std_bc,logistic_regression_dt_std_bc, file="models_DT_STD_BC.Rdata")
load("models_DT_STD_BC.Rdata")

#Modelling on Lasso + STD + bc
# random_forest_lasso_std_bc <- train(is_fake~., data=train_lasso_std_bc, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_lasso_std_bc <- train(is_fake ~ ., data = train_lasso_std_bc, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                         trControl = control)
# xgboost_model_lasso_std_bc <- train(is_fake ~ ., data = train_lasso_std_bc, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# logistic_regression_lasso_std_bc <- train(is_fake ~ ., data = train_lasso_std_bc, method = "glm",family = "binomial",trControl = control,metric=metric)
# ann_model_lasso_std_bc <- train(is_fake ~ ., data = train_lasso_std_bc, method = "nnet",trControl = control,metric="ROC", trace = FALSE, linout = FALSE, maxit = 400)
# save(random_forest_lasso_std_bc,decision_tree_lasso_std_bc,xgboost_model_lasso_std_bc,logistic_regression_lasso_std_bc, ann_model_lasso_std_bc, file="models_Lasso_STD_BC.Rdata")
load("models_Lasso_STD_BC.Rdata")

#MODELLING only on raw train data
# random_forest_raw <- train(is_fake~., data=train_std, method="rf", metric=metric, ntree=250, trControl=control)
# decision_tree_raw <- train(is_fake ~ ., data = train_std, method = "rpart",metric=metric,
#                         tuneLength = 10,
#                         trControl = control)
# xgboost_model_raw <- train(is_fake ~ ., data = train_std, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)
# logistic_regression_raw <- train(is_fake ~ ., data = train_std, method = "glm",family = "binomial",trControl = control,metric=metric,preProcess = c("center", "scale"))
# ann_model_raw <- train(is_fake ~ ., data = train_std, method = "nnet",trControl = control,metric="ROC",preProcess = c("center", "scale"), trace = FALSE, linout = FALSE, maxit = 200)
# save(random_forest_raw,decision_tree_raw,xgboost_model_raw,logistic_regression_raw,ann_model_raw, file="models_Raw_NoPreProc.Rdata")
load("models_Raw_NoPreProc.Rdata")

#Model Assessment
#We will create a list of all the models
model_list <- list(
  random_forest,
  decision_tree,
  xgboost_model,
  logistic_regression,
  ann_model,
  
  random_forest_lasso,
  decision_tree_lasso,
  xgboost_model_lasso,
  logistic_regression_lasso,
  ann_model_lasso,
  
  random_forest_dt_std,
  decision_tree_dt_std,
  xgboost_model_dt_std,
  
  random_forest_lasso_std,
  decision_tree_lasso_std,
  xgboost_model_lasso_std,
  
  random_forest_dt_std_bc,
  decision_tree_dt_std_bc,
  xgboost_model_dt_std_bc,
  logistic_regression_lasso_std_bc,
  ann_model_dt_std_bc,
  
  random_forest_lasso_std_bc,
  decision_tree_lasso_std_bc,
  xgboost_model_lasso_std_bc,
  logistic_regression_lasso_std_bc,
  ann_model_lasso_std_bc,
  
  random_forest_raw,
  decision_tree_raw,
  xgboost_model_raw,
  logistic_regression_raw,
  ann_model_raw
)
model_names <- c(
  "RF", "DT", "XGB", "LR", "ANN",
  
  "RF_Lasso", "DT_Lasso", "XGB_Lasso", "LR_Lasso", "ANN_Lasso",
  
  "RF_DT_STD", "DT_DT_STD", "XGB_DT_STD",
  
  "RF_Lasso_STD", "DT_Lasso_STD", "XGB_Lasso_STD",
  
  "RF_DT_STD_BC", "DT_DT_STD_BC", "XGB_DT_STD_BC", "LR_Lasso_STD_BC", "ANN_DT_STD_BC",
  
  "RF_Lasso_STD_BC", "DT_Lasso_STD_BC", "XGB_Lasso_STD_BC", "LR_Lasso_STD_BC", "ANN_Lasso_STD_BC",
  
  "RF_Raw", "DT_Raw", "XGB_Raw", "LR_Raw", "ANN_Raw"
)

results <- resamples(model_list, modelNames=model_names)
bwplot(results, 
       horizontal = TRUE,           # Modelli verticali, metriche orizzontali
       layout = c(3, 1),            # 3 grafici in riga (ROC | Sens | Spec)
       scales = list(x = list(relation = "free")),  # Scale indipendenti
       panel = function(...) {      # Custom panel per sorting
         panel.bwplot(...)
       })
summary(results)

#ROC for EVERY Model Category 

test_dt$RF=predict(random_forest,test_dt, type="prob")[,2]
test_dt$XGboost=predict(xgboost_model,test_dt, "prob")[,2]
test_dt$logistic=predict(logistic_regression,test_dt, "prob")[,2]
test_dt$dt=predict(decision_tree,test_dt, "prob")[,2]
test_dt$ann=predict(ann_model,test_dt, "prob")[,2]

library(pROC)
# See roc values ########
roc.random_forest=roc(is_fake ~ RF, data = test_dt)
roc.xgboost_model=roc(is_fake ~ XGboost, data = test_dt)
roc.logistic_regression=roc(is_fake ~ logistic, data = test_dt)
roc.decision_tree=roc(is_fake ~ dt, data = test_dt)
roc.ann_model=roc(is_fake ~ ann, data = test_dt)

plot(roc.random_forest)
plot(roc.xgboost_model,add=T,col="red")
plot(roc.logistic_regression,add=T,col="blue")
plot(roc.decision_tree,add=T,col="yellow")
plot(roc.ann_model,add=T,col="green")

#DT no preprocess non si capisce bene....
library(funModeling)
gain_lift(data = test_dt, score = 'RF', target = 'is_fake')
gain_lift(data = test_dt, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_dt, score = 'dt', target = 'is_fake') # Best for now 

#ROC for Lasso No Pre Pro
test_lasso$RF=predict(random_forest_lasso,test_lasso,type="prob")[,2]
test_lasso$XGboost=predict(xgboost_model_lasso,test_lasso,type="prob")[,2]
test_lasso$logistic=predict(logistic_regression_lasso,test_lasso,type="prob")[,2]
test_lasso$dt=predict(decision_tree_lasso,test_lasso,type="prob")[,2]
test_lasso$ann=predict(ann_model_lasso,test_lasso,type="prob")[,2]

roc.random_forest_lasso=roc(is_fake ~ RF, data = test_lasso)
roc.xgboost_model_lasso=roc(is_fake ~ XGboost, data = test_lasso)
roc.logistic_regression_lasso=roc(is_fake ~ logistic, data = test_lasso)
roc.decision_tree_lasso=roc(is_fake ~ dt, data = test_lasso)
roc.ann_mode_lassol=roc(is_fake ~ ann, data = test_lasso)

plot(roc.random_forest_lasso)
plot(roc.xgboost_model_lasso,add=T,col="red")
plot(roc.logistic_regression_lasso,add=T,col="blue")
plot(roc.decision_tree_lasso,add=T,col="yellow")
plot(roc.ann_mode_lassol,add=T,col="green")

gain_lift(data = test_lasso, score = 'RF', target = 'is_fake') #Best here
gain_lift(data = test_lasso, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_lasso, score = 'dt', target = 'is_fake')
gain_lift(data = test_lasso, score = 'ann', target = 'is_fake')

#ROC for DT + STD
test_dt_std$RF=predict(random_forest_dt_std,test_dt_std,type="prob")[,2]
test_dt_std$XGboost=predict(xgboost_model_dt_std,test_dt_std,type="prob")[,2]
test_dt_std$dt=predict(decision_tree_dt_std,test_dt_std,type="prob")[,2]

roc.random_forest_dt_std=roc(is_fake ~ RF, data = test_dt_std)
roc.xgboost_model_dt_std=roc(is_fake ~ XGboost, data = test_dt_std)
roc.decision_tree_dt_std=roc(is_fake ~ dt, data = test_dt_std)


plot(roc.random_forest_dt_std)
plot(roc.xgboost_model_dt_std,add=T,col="red")
plot(roc.decision_tree_dt_std,add=T,col="yellow")

gain_lift(data = test_dt_std, score = 'RF', target = 'is_fake') 
gain_lift(data = test_dt_std, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_dt_std, score = 'dt', target = 'is_fake') #Best

#ROC for Lasso + STD

test_lasso_std$RF=predict(random_forest_lasso_std,test_lasso_std,type="prob")[,2]
test_lasso_std$XGboost=predict(xgboost_model_lasso_std,test_lasso_std,type="prob")[,2]
test_lasso_std$dt=predict(decision_tree_lasso_std,test_lasso_std,type="prob")[,2]


roc.random_forest_lasso_std=roc(is_fake ~ RF, data = test_lasso_std)
roc.xgboost_model_lasso_std=roc(is_fake ~ XGboost, data = test_lasso_std)
roc.decision_tree_lasso_std=roc(is_fake ~ dt, data = test_lasso_std)

plot(roc.random_forest_lasso_std)
plot(roc.xgboost_model_lasso_std,add=T,col="red")
plot(roc.decision_tree_lasso_std,add=T,col="yellow")

gain_lift(data = test_lasso_std, score = 'RF', target = 'is_fake') #Best Here
gain_lift(data = test_lasso_std, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_lasso_std, score = 'dt', target = 'is_fake')

#ROC for DT + STD+bc
test_dt_std_bc$RF=predict(random_forest_dt_std_bc,test_dt_std_bc,type="prob")[,2]
test_dt_std_bc$XGboost=predict(xgboost_model_dt_std_bc,test_dt_std_bc,type="prob")[,2]
test_dt_std_bc$dt=predict(decision_tree_dt_std_bc,test_dt_std_bc,type="prob")[,2]
test_dt_std_bc$ann=predict(ann_model_dt_std_bc,test_dt_std_bc,type="prob")[,2]
test_dt_std_bc$logistic=predict(logistic_regression_dt_std_bc,test_dt_std_bc,type="prob")[,2]

roc.random_forest_dt_std_bc=roc(is_fake ~ RF, data = test_dt_std_bc)
roc.xgboost_model_dt_std_bc=roc(is_fake ~ XGboost, data = test_dt_std_bc)
roc.decision_tree_dt_std_bc=roc(is_fake ~ dt, data = test_dt_std_bc)
roc.logistic_regression_dt_std_bc = roc(is_fake ~ logistic, data = test_dt_std_bc)
roc.ann_model_dt_std_bc = roc(is_fake ~ ann, data = test_dt_std_bc)

plot(roc.random_forest_dt_std_bc)
plot(roc.xgboost_model_dt_std_bc,add=T,col="red")
plot(roc.logistic_regression_dt_std_bc,add=T,col="blue")
plot(roc.decision_tree_dt_std_bc,add=T,col="yellow")
plot(roc.ann_model_dt_std_bc,add=T,col="green")

gain_lift(data = test_dt_std_bc, score = 'RF', target = 'is_fake') 
gain_lift(data = test_dt_std_bc, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_dt_std_bc, score = 'dt', target = 'is_fake') #Best here
gain_lift(data = test_dt_std_bc, score = 'ann', target = 'is_fake')

#ROC for Lasso + STD + bc

test_lasso_std_bc$RF=predict(random_forest_lasso_std_bc,test_lasso_std_bc,type="prob")[,2]
test_lasso_std_bc$XGboost=predict(xgboost_model_lasso_std_bc,test_lasso_std_bc,type="prob")[,2]
test_lasso_std_bc$dt=predict(decision_tree_lasso_std_bc,test_lasso_std_bc,type="prob")[,2]
test_lasso_std_bc$ann=predict(ann_model_lasso_std_bc,test_lasso_std_bc,type="prob")[,2]
test_lasso_std_bc$logistic=predict(logistic_regression_lasso_std_bc,test_lasso_std_bc,type="prob")[,2]

roc.random_forest_lasso_std_bc=roc(is_fake ~ RF, data = test_lasso_std_bc)
roc.xgboost_model_lasso_std_bc=roc(is_fake ~ XGboost, data = test_lasso_std_bc)
roc.decision_tree_lasso_std_bc=roc(is_fake ~ dt, data = test_lasso_std_bc)
roc.logistic_regression_lasso_std_bc=roc(is_fake  ~ logistic, data = test_lasso_std_bc)
roc.ann_model_lasso_std_bc = roc(is_fake  ~ ann, data = test_lasso_std_bc)

plot(roc.random_forest_lasso_std_bc)
plot(roc.xgboost_model_lasso_std_bc,add=T,col="red")
plot(roc.decision_tree_lasso_std_bc,add=T,col="yellow")
plot(roc.ann_model_lasso_std_bc,add=T,col="green")
plot(roc.logistic_regression_lasso_std_bc,add=T,col="blue")

gain_lift(data = test_lasso_std_bc, score = 'RF', target = 'is_fake') #Best Here
gain_lift(data = test_lasso_std_bc, score = 'XGboost', target = 'is_fake')
gain_lift(data = test_lasso_std_bc, score = 'dt', target = 'is_fake')

#ROC for Raw 

test$RF=predict(random_forest_raw,test,type="prob")[,2]
test$XGboost=predict(xgboost_model_raw,test,type="prob")[,2]
test$dt=predict(decision_tree_raw,test,type="prob")[,2]
test$ann=predict(ann_model_raw,test,type="prob")[,2]
test$logistic=predict(logistic_regression_raw,test,type="prob")[,2]


roc.random_forest_raw=roc(is_fake ~ RF, data = test)
roc.xgboost_model_raw=roc(is_fake ~ XGboost, data = test)
roc.decision_tree_raw=roc(is_fake ~ dt, data = test)
roc.logistic_regression_raw=roc(is_fake  ~ logistic, data = test)
roc.ann_model_raw = roc(is_fake  ~ ann, data = test)

plot(roc.random_forest_raw)
plot(roc.xgboost_model_raw,add=T,col="red")
plot(roc.decision_tree_raw,add=T,col="yellow")
plot(roc.ann_model_raw,add=T,col="green")
plot(roc.logistic_regression_raw,add=T,col="blue")

#Tutti schifo rispetto altri 
#Best Model RF + Lasso + STD

df = test_lasso_std
df$lasso <- NULL
df$XGboost <- NULL# winner model

library(dplyr)
# for each threshold, find tp, tn, fp, fn and the sens=prop_true_M, spec=prop_true_R, precision=tp/(tp+fp)
thresholds <- seq(from = 0, to = 1, by = 0.01)
prop_table <- data.frame(threshold = thresholds, prop_true_M = NA,  prop_true_R = NA, true_M = NA,  true_R = NA ,fn_M=NA)
for (threshold in thresholds) {
  pred <- ifelse(df$RF > threshold, "one", "zero")  # be careful here!!!
  pred_t <- ifelse(pred == df$is_fake, TRUE, FALSE)
  
  group <- data.frame(df, "pred" = pred_t) %>%
    group_by(is_fake, pred) %>%
    dplyr::summarise(n = n())
  
  group_M <- filter(group, is_fake == "one")
  
  true_M=sum(filter(group_M, pred == TRUE)$n)
  prop_M <- sum(filter(group_M, pred == TRUE)$n) / sum(group_M$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_M"] <- prop_M
  prop_table[prop_table$threshold == threshold, "true_M"] <- true_M
  
  fn_M=sum(filter(group_M, pred == FALSE)$n)
  # true M predicted as R
  prop_table[prop_table$threshold == threshold, "fn_M"] <- fn_M
  
  
  group_R <- filter(group, is_fake == "zero")
  
  true_R=sum(filter(group_R, pred == TRUE)$n)
  prop_R <- sum(filter(group_R, pred == TRUE)$n) / sum(group_R$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_R"] <- prop_R
  prop_table[prop_table$threshold == threshold, "true_R"] <- true_R
  
}


head(prop_table, n=10)
confusionMatrix(
  data = factor(ifelse(df$RF > 0.2, "one", "zero"), levels = c("zero", "one")),  # Forza ordine
  reference = factor(df$is_fake, levels = c("zero", "one")),
  positive = "one")

# calculate other missing measures

# n of observations of the validation set    
prop_table$n=nrow(df)

# false positive (fp_M) by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_M=nrow(df)-prop_table$true_R-prop_table$true_M-prop_table$fn_M

# find accuracy
prop_table$acc=(prop_table$true_R+prop_table$true_M)/nrow(df)

# find precision
prop_table$prec_M=prop_table$true_M/(prop_table$true_M+prop_table$fp_M)

# find F1 =2*(prec*sens)/(prec+sens)
# prop_true_M = sensitivity

prop_table$F1=2*(prop_table$prop_true_M*prop_table$prec_M)/(prop_table$prop_true_M+prop_table$prec_M)

# verify not having NA metrics at start or end of data 
tail(prop_table)
head(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_M=impute(prop_table$prec_M, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table)
colnames(prop_table)
# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
head(prop_table2)
head(prop_table)
# plot measures vs soglia
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)
gathered=prop_table2 %>%
  gather(x, y, prop_true_M:F1)
head(gathered)


##########################################################################
# grafico con tutte le misure 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")

# follow sensitivity= prop true M...beccre i veri ricchi (soglie basse) 
# anche F1 ciferma soglie attorno a  0.20
confusionMatrix(
  data = factor(ifelse(df$RF > 0.2, "one", "zero"), levels = c("zero", "one")),  # Forza ordine
  reference = factor(df$is_fake, levels = c("zero", "one")),
  positive = "one")

#
