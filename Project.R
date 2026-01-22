setwd("/Users/marcelpotinga/Downloads/archive (1)")
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
summary(logitTuneCvA)

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

# metric <- "ROC"
# control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE,savePrediction = TRUE)
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
# logistic_regression_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "glm",family = "binomial",trControl = control,metric=metric)
# ann_model_dt_std_bc <- train(is_fake ~ ., data = train_dt_std_bc, method = "nnet",trControl = control,metric="ROC", trace = FALSE, linout = FALSE, maxit = 200)
# save(random_forest_dt_std_bc,decision_tree_dt_std_bc,xgboost_model_dt_std_bc, file="models_DT_STD_BC.Rdata")
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

results <- resamples(model_list, modelNames=model_names)
bwplot(results, 
       horizontal = TRUE,           # Modelli verticali, metriche orizzontali
       layout = c(3, 1),            # 3 grafici in riga (ROC | Sens | Spec)
       scales = list(x = list(relation = "free")),  # Scale indipendenti
       panel = function(...) {      # Custom panel per sorting
         panel.bwplot(...)
       })
cat("Lasso solo:", max(xgboost_model_lasso$results$Sens), "\n")
cat("XGB post-Lasso:", max(xgboost_model$results$Sens), "\n")


preds_class_lasso <- predict(xgboost_model_lasso, test_lasso, type = "raw")  # Factor classi
#Confusion matrix Lasso
conf_mat <- confusionMatrix(preds_class, test_lasso$is_fake, 
                            positive = "one")

preds_class_dt <- predict(xgboost_model, test_dt, type = "raw")  # Factor classi
#Confusion matrix XGB DT
conf_mat <- confusionMatrix(preds_class, test_dt$is_fake, 
                            positive = "one")  # "one"=fake
print(conf_mat)

#We will compare LIFT Charts
library(funModeling)
probs_xgb <- predict(xgboost_model, test_dt, type = "prob")[, "one"]
test_dt$postXGB <- probs_xgb 
xgb_lift <- gain_lift(data = test_dt, score = 'postXGB', target = 'is_fake')
plot(xgb_lift)  # Lift curve
print(xgb_lift)

#Lasso Lift Chart
probs_xgb_lasso <- predict(xgboost_model_lasso, test_lasso, type = "prob")[, "one"]
test_lasso$postXGB <- probs_xgb_lasso
xgb_lift_lasso <- gain_lift(data = test_lasso, score = 'postXGB', target = 'is_fake')
plot(xgb_lift_lasso)  # Lift curve
print(xgb_lift_lasso)

#XGB lasso winning model NO scaling on variables.
probs <- test_lasso$postXGB
roc_obj <- roc(test_lasso$is_fake, probs, 
               levels = c("zero", "one"), 
               direction = "<", quiet = TRUE)  
coords(roc_obj, "best", ret=c("threshold", "sensitivity", "specificity"))

#PLOT ROC
plot(roc_obj, 
     main = "ROC Curve - XGB Lasso (4 feat)", 
     print.auc = TRUE,           # AUC in legenda
     print.thres = "best",       # Soglia Youden ottimale
     col = "blue", lwd = 2)


my_threshold <- 0.359  # O 0.8, 0.9, ecc.

#Predictions
preds_custom <- ifelse(probs > my_threshold, "one", "zero")
conf_custom <- confusionMatrix(
  factor(preds_custom, levels = c("zero", "one")), 
  factor(test_lasso$is_fake, levels = c("zero", "one")), 
  positive = "one"  # "one" = fake
)

# 5. Stampa risultati chiave
print(conf_custom)

