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

metric <- "Spec"
control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE,savePrediction = TRUE)
random_forest <- train(is_fake~., data=train_dt, method="rf", metric=metric, ntree=250, trControl=control)

decision_tree <- train(is_fake ~ ., data = train_dt, method = "rpart",metric=metric,
      tuneLength = 10,
      trControl = control)

xgboost_model <- train(is_fake ~ ., data = train_dt, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)

logistic_regression <- train(is_fake ~ ., data = train_dt, method = "glm",family = "binomial",trControl = control,metric=metric,preProcess = c("center", "scale")
)

library(nnet)
ann_model <- train(is_fake ~ ., data = train_dt, method = "nnet",trControl = control,metric=metric,preProcess = c("center", "scale"), trace = FALSE, linout = FALSE, maxit = 200)
save(random_forest,decision_tree,xgboost_model,logistic_regression,ann_model, file="models_DT_NoPreProc.Rdata")

#Modelling on Lasso MS no SCALING
random_forest_lasso <- train(is_fake~., data=train_lasso, method="rf", metric=metric, ntree=250, trControl=control)

decision_tree_lasso <- train(is_fake ~ ., data = train_lasso, method = "rpart",metric=metric,
                       tuneLength = 10,
                       trControl = control)

xgboost_model_lasso <- train(is_fake ~ ., data = train_lasso, method = "xgbTree", trControl = control, metric = metric, use_rmm=TRUE, verbose=0)

logistic_regression_lasso <- train(is_fake ~ ., data = train_lasso, method = "glm",family = "binomial",trControl = control,metric=metric,preProcess = c("center", "scale")
)

library(nnet)
ann_model_lasso <- train(is_fake ~ ., data = train_lasso, method = "nnet",trControl = control,metric=metric,preProcess = c("center", "scale"), trace = FALSE, linout = FALSE, maxit = 200)
save(random_forest_lasso,decision_tree_lasso,xgboost_model_lasso,logistic_regression_lasso,ann_model_lasso, file="models_Lasso_NoPreProc.Rdata")
