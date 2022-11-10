# Regularized logistic regression *WITHOUT* Gender as a predictor variable

# SECTION §1.0 -- Pre-requisite libraries/packages ----

# Necessary libraries required for code to function 
library(readr)     # for importing data into RStudio
library(caret)     # for functions required to generate predictive models
library(ROCR)      # for ROC curve analysis code 
library(pROC)      # for ROC curve analysis code
library(LiblineaR) # for regularized logistic regression

# SECTION §2.0 -- code used to import the AOSI Training, Testing, and Independent data into R ----
AOSI.training <- read_csv("03_Data/02_AOSI Machine Learning Datasets (Train-Test 80-20)/dx36 transformed to 0 and 1 for logistic regression/AOSI.training set Sept 15 2022_transformed.csv")
AOSI.testing <- read_csv("03_Data/02_AOSI Machine Learning Datasets (Train-Test 80-20)/dx36 transformed to 0 and 1 for logistic regression/AOSI.testing set Sept 15 2022_transformed.csv")
AOSI.independent <- read_csv("03_Data/03_AOSI new ASIB independent dataset/dx36 transformed to 0 and 1/AOSI.independent - dx36 transformed.csv")

# treat 36-month diagnostic outcome (0 = IL-ASD, 1 = N) as a factor.
AOSI.training$dx36 <- as.factor(AOSI.training$dx36)
AOSI.testing$dx36 <- as.factor(AOSI.testing$dx36)
AOSI.independent$dx36 <- as.factor(AOSI.independent$dx36)

# changes dx36 outcome in the Training and Test sets from 0 & 1 to ASD and N respectively
levels(AOSI.training$dx36) <- c("ASD", "N")
levels(AOSI.testing$dx36) <- c("ASD", "N")
levels(AOSI.independent$dx36) <- c("ASD", "N")

# SECTION §3.0 -- generating custom summary and training control functions for regression modelling in caret ----

# this code generates a custom summary function that allows for a more detailed
# model performance metric report. It was sourced from a comment on stackoverflow; 
#  https://stackoverflow.com/questions/52691761/additional-metrics-in-caret-ppv-sensitivity-specificity 

Custom.MySummary  <- function(data, lev = NULL, model = NULL){
  a1 <- defaultSummary(data, lev, model)     # outputs accuracy and Kappa
  b1 <- twoClassSummary(data, lev, model)    # outputs area under the ROC curve, sensitivity, specificity 
  c1 <- prSummary(data, lev, model)          # outputs precision (PPV) and recall (NPV)
  out <- c(a1, b1, c1)
  out}

# this code takes the customized summary function and puts it into a custom trControl function we can call when training the predictive models 
Custom.Ctrl <- trainControl(method = "cv",                         # the re-sampling method being used 
                            number = 10,                           # the number/fold of re-sampling iterations,
                            savePredictions = TRUE,                # saves model predictions
                            summaryFunction = Custom.MySummary,    # calls the custom summary function defined above
                            classProbs = TRUE)              

# SECTION §4.0 -- generating regularized logistic regression models ---- 

# REGARDING WARNING MESSAGES
# NOTE 1: The code  may output a warning message (see below) that can be discounted;
# the warning almost always pops up because some tuning parameter combination produced
# predictions that are constant for all samples. Caret's train() function tries 
# to compute the R^2 and, since it needs a non-zero variance, it can produce NA 
# for that statistic. See https://github.com/topepo/caret/issues/1124 for more detail

# In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
#                          There were missing values in resampled performance measures.


# NOTE 2: Training some of the logistic regression models can result in a warning 
# message (see below).This can stem from (1) two predictor variables being perfectly 
# correlated, or (2) there aare more model parameters than observations in the 
# dataset. Given that there is *definitely* not more model parameters than observations
# (AOSI item-level + Total score = 17; MSEL = 5; 22 parameters vs a training set
# with 300+ observations) it is likely the perfectly correlated issue which is 
# entirely plausible given AOSI scoring data. See the link below for more info
# https://www.statology.org/prediction-from-rank-deficient-fit-may-be-misleading/

# In predict.lm(object, newdata, se.fit, scale = 1, type = if (type ==  ... :
# prediction from a rank-deficient fit may be misleading


# MODEL 1: Item-level data
set.seed(17724)
cv_model_01 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_01, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_01.rda")

# MODEL 2: Item-level data + AOSI Total Score 
set.seed(17724)
cv_model_02 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_02, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_02.rda")

# MODEL 3: Item-level data + MSEL subscales
set.seed(17724)
cv_model_03 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_03, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_03.rda")

# MODEL 4: Item-level data + Total Score + MSEL subscales
set.seed(17724)
cv_model_04 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + AQTS + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_04, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_04.rda")

# MODEL 5: Total Score + MSEL subscales 
set.seed(17724)
cv_model_05 <- train(
  dx36 ~ AQTS + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_05, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_05.rda")

# MODEL 6: MSEL subscales
set.seed(17724)
cv_model_06 <- train(
  dx36 ~ MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_06, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_06.rda")

# MODEL 7: Total score 
set.seed(17724)
cv_model_07 <- train(
  dx36 ~ AQTS, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_07, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_07.rda")

# MODEL 8: Factor analysis items 
set.seed(17724)
cv_model_08 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_08, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_08.rda")

# MODEL 9: Factor analysis items + Total Score
set.seed(17724)
cv_model_09 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_09, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_09.rda")

# MODEL 10: Factor analysis items + MSEL subscales 
set.seed(17724)
cv_model_10 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_10, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_10.rda")

# MODEL 11: Factor analysis items + Total Score + MSEL subscales
set.seed(17724)
cv_model_11 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + AQTS + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_11, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_11.rda")

# MODEL 12: Factor  analysis items surviving post hoc comparisons
set.seed(17724)
cv_model_12 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_12, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_12.rda")

# MODEL 13: Factor  analysis items surviving post hoc comparisons + Total Score
set.seed(17724)
cv_model_13 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_13, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_13.rda")

# MODEL 14: Factor  analysis items surviving post hoc comparisons + MSEL subscales 
set.seed(17724)
cv_model_14 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss,
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_14, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_14.rda")

# MODEL 15: Factor analysis items surviving post hoc comparisons + Total SCore + MSEL subscales 
set.seed(17724)
cv_model_15 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + AQTS + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss,
  data = AOSI.training, 
  method = "regLogistic",
  metric = "ROC",
  trControl = Custom.Ctrl,
  tuneLength = 10
)
save(cv_model_15, file = "RegLogistic TL10 WITHOUT gender Oct 03 cv_model_15.rda")

# SECTION §5.0 -- model performance (ROC curve auc, sens, spec, ppv, npv) ----

# Generating vectors to store model statistics (when the model is applied to *TEST* data) to generate a data table for ease of recording/interpreting results
{ auc.vector.test <- c("AUC")
  number.vector.test <- c("Model#")
  accuracy.vector.test <- c("Accuracy")
  accuracy.99CI.lower.vector.test <- c("99% CI lower bound")
  accuracy.99CI.upper.vector.test <- c("99% CI upper bound")
  Kappa.vector.test <- c("Kappa")
  McnemarP.vector.test <- c("McnemarPvalue")
  sensitivity.vector.test <- c("Sensitivity")
  specificity.vector.test <- c("Specificity")
  PPV.vector.test <- c("PPV")
  NPV.vector.test <- c("NPV")
  precision.vector.test <- c("Precision")
  recall.vector.test <- c("Recall")}

# Generating vectors to store model statistics (when the model is applied to *TRAINING* data) to generate a data table for ease of recording/interpreting results
{ auc.vector.train <- c("AUC")
  number.vector.train <- c("Model#")
  accuracy.vector.train <- c("Accuracy")
  accuracy.99CI.lower.vector.train <- c("99% CI lower bound")
  accuracy.99CI.upper.vector.train <- c("99% CI upper bound")
  Kappa.vector.train <- c("Kappa")
  McnemarP.vector.train <- c("McnemarPvalue")
  sensitivity.vector.train <- c("Sensitivity")
  specificity.vector.train <- c("Specificity")
  PPV.vector.train <- c("PPV")
  NPV.vector.train <- c("NPV")
  precision.vector.train <- c("Precision")
  recall.vector.train <- c("Recall")}

# Generating vectors to store model statistics (when the model is applied to *INDEPENDENT* test data) to generate a data table for ease of recording/interpreting results
{ auc.vector.independent <- c("AUC")
  number.vector.independent <- c("Model#")
  accuracy.vector.independent <- c("Accuracy")
  accuracy.99CI.lower.vector.independent <- c("99% CI lower bound")
  accuracy.99CI.upper.vector.independent <- c("99% CI upper bound")
  Kappa.vector.independent <- c("Kappa")
  McnemarP.vector.independent <- c("McnemarPvalue")
  sensitivity.vector.independent <- c("Sensitivity")
  specificity.vector.independent <- c("Specificity")
  PPV.vector.independent <- c("PPV")
  NPV.vector.independent <- c("NPV")
  precision.vector.independent <- c("Precision")
  recall.vector.independent <- c("Recall")}


# MODEL 01 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs01.proc.test <- predict(cv_model_01, newdata=AOSI.testing, type="prob")
  rocCurve01.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs01.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc01.test <- auc(rocCurve01.proc.test)
  
  # please note that the code below is an alternate way of calculating AUC. The only
  # difference is that the code below uses the R library ROCR instead of the R 
  # library pROC used in the code above. Both approaches end up calculating the 
  # same AUC values. The pROC code was selected in this application because it is 
  # slightly shorter and uses less intermediary variables. 
  
  #prob01.test <- predict(cv_model_01, newdata=AOSI.testing, type="prob") [,2] 
  #pred01.test <- prediction(prob01.test, AOSI.testing$dx36)
  #perf01.test <- performance(pred01.test, measure = "tpr", x.measure = "fpr")
  #plot(perf01.test)
  #auc01.test <- performance(pred01.test, measure = "auc")
  #auc01.test <- auc01.test@y.values[[1]]
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class01.test <- predict(cv_model_01, AOSI.testing)
  test.confusionMatrix01 <- confusionMatrix(
    data = relevel(pred_class01.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc01.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 01")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix01[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix01[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs01.proc.test, rocCurve01.proc.test, auc01.test, pred_class01.test, test.confusionMatrix01)}
# calculate and extract model performance on training set (AOSI.training)
{ probs01.proc.train <- predict(cv_model_01, newdata=AOSI.training, type="prob")
  rocCurve01.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs01.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc01.train <- auc(rocCurve01.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class01.train <- predict(cv_model_01, AOSI.training)
  train.confusionMatrix01 <- confusionMatrix(
    data = relevel(pred_class01.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc01.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 01")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix01[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix01[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs01.proc.train, rocCurve01.proc.train, auc01.train, pred_class01.train, train.confusionMatrix01)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent01.proc <- predict(cv_model_01, newdata=AOSI.independent, type="prob")
  rocCurve01.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent01.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc01.independent <- auc(rocCurve01.proc)
  
  pred_class01 <- predict(cv_model_01, AOSI.independent)
  independent.confusionMatrix01 <- confusionMatrix(
    data = relevel(pred_class01, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc01.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 01")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix01[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix01[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent01.proc, rocCurve01.proc, auc01.independent, pred_class01, independent.confusionMatrix01)}


# MODEL 02 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs02.proc.test <- predict(cv_model_02, newdata=AOSI.testing, type="prob")
  rocCurve02.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs02.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc02.test <- auc(rocCurve02.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class02.test <- predict(cv_model_02, AOSI.testing)
  test.confusionMatrix02 <- confusionMatrix(
    data = relevel(pred_class02.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc02.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 02")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix02[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix02[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs02.proc.test, rocCurve02.proc.test, auc02.test, pred_class02.test, test.confusionMatrix02)}
# calculate and extract model performance on training set (AOSI.training)
{ probs02.proc.train <- predict(cv_model_02, newdata=AOSI.training, type="prob")
  rocCurve02.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs02.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc02.train <- auc(rocCurve02.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class02.train <- predict(cv_model_02, AOSI.training)
  train.confusionMatrix02 <- confusionMatrix(
    data = relevel(pred_class02.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc02.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 02")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix02[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix02[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs02.proc.train, rocCurve02.proc.train, auc02.train, pred_class02.train, train.confusionMatrix02)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent02.proc <- predict(cv_model_02, newdata=AOSI.independent, type="prob")
  rocCurve02.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent02.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc02.independent <- auc(rocCurve02.proc)
  
  pred_class02 <- predict(cv_model_02, AOSI.independent)
  independent.confusionMatrix02 <- confusionMatrix(
    data = relevel(pred_class02, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc02.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 02")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix02[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix02[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent02.proc, rocCurve02.proc, auc02.independent, pred_class02, independent.confusionMatrix02)}


# MODEL 03 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs03.proc.test <- predict(cv_model_03, newdata=AOSI.testing, type="prob")
  rocCurve03.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs03.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc03.test <- auc(rocCurve03.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class03.test <- predict(cv_model_03, AOSI.testing)
  test.confusionMatrix03 <- confusionMatrix(
    data = relevel(pred_class03.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc03.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 03")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix03[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix03[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs03.proc.test, rocCurve03.proc.test, auc03.test, pred_class03.test, test.confusionMatrix03)}
# calculate and extract model performance on training set (AOSI.training)
{ probs03.proc.train <- predict(cv_model_03, newdata=AOSI.training, type="prob")
  rocCurve03.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs03.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc03.train <- auc(rocCurve03.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class03.train <- predict(cv_model_03, AOSI.training)
  train.confusionMatrix03 <- confusionMatrix(
    data = relevel(pred_class03.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc03.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 03")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix03[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix03[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs03.proc.train, rocCurve03.proc.train, auc03.train, pred_class03.train, train.confusionMatrix03)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent03.proc <- predict(cv_model_03, newdata=AOSI.independent, type="prob")
  rocCurve03.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent03.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc03.independent <- auc(rocCurve03.proc)
  
  pred_class03 <- predict(cv_model_03, AOSI.independent)
  independent.confusionMatrix03 <- confusionMatrix(
    data = relevel(pred_class03, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc03.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 03")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix03[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix03[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent03.proc, rocCurve03.proc, auc03.independent, pred_class03, independent.confusionMatrix03)}


# MODEL 04 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs04.proc.test <- predict(cv_model_04, newdata=AOSI.testing, type="prob")
  rocCurve04.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs04.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc04.test <- auc(rocCurve04.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class04.test <- predict(cv_model_04, AOSI.testing)
  test.confusionMatrix04 <- confusionMatrix(
    data = relevel(pred_class04.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc04.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 04")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix04[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix04[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs04.proc.test, rocCurve04.proc.test, auc04.test, pred_class04.test, test.confusionMatrix04)}
# calculate and extract model performance on training set (AOSI.training)
{ probs04.proc.train <- predict(cv_model_04, newdata=AOSI.training, type="prob")
  rocCurve04.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs04.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc04.train <- auc(rocCurve04.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class04.train <- predict(cv_model_04, AOSI.training)
  train.confusionMatrix04 <- confusionMatrix(
    data = relevel(pred_class04.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc04.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 04")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix04[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix04[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs04.proc.train, rocCurve04.proc.train, auc04.train, pred_class04.train, train.confusionMatrix04)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent04.proc <- predict(cv_model_04, newdata=AOSI.independent, type="prob")
  rocCurve04.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent04.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc04.independent <- auc(rocCurve04.proc)
  
  pred_class04 <- predict(cv_model_04, AOSI.independent)
  independent.confusionMatrix04 <- confusionMatrix(
    data = relevel(pred_class04, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc04.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 04")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix04[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix04[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent04.proc, rocCurve04.proc, auc04.independent, pred_class04, independent.confusionMatrix04)}


# MODEL 05 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs05.proc.test <- predict(cv_model_05, newdata=AOSI.testing, type="prob")
  rocCurve05.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs05.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc05.test <- auc(rocCurve05.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class05.test <- predict(cv_model_05, AOSI.testing)
  test.confusionMatrix05 <- confusionMatrix(
    data = relevel(pred_class05.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc05.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 05")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix05[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix05[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs05.proc.test, rocCurve05.proc.test, auc05.test, pred_class05.test, test.confusionMatrix05)}
# calculate and extract model performance on training set (AOSI.training)
{ probs05.proc.train <- predict(cv_model_05, newdata=AOSI.training, type="prob")
  rocCurve05.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs05.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc05.train <- auc(rocCurve05.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class05.train <- predict(cv_model_05, AOSI.training)
  train.confusionMatrix05 <- confusionMatrix(
    data = relevel(pred_class05.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc05.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 05")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix05[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix05[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs05.proc.train, rocCurve05.proc.train, auc05.train, pred_class05.train, train.confusionMatrix05)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent05.proc <- predict(cv_model_05, newdata=AOSI.independent, type="prob")
  rocCurve05.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent05.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc05.independent <- auc(rocCurve05.proc)
  
  pred_class05 <- predict(cv_model_05, AOSI.independent)
  independent.confusionMatrix05 <- confusionMatrix(
    data = relevel(pred_class05, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc05.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 05")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix05[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix05[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent05.proc, rocCurve05.proc, auc05.independent, pred_class05, independent.confusionMatrix05)}


# MODEL 06 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs06.proc.test <- predict(cv_model_06, newdata=AOSI.testing, type="prob")
  rocCurve06.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs06.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc06.test <- auc(rocCurve06.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class06.test <- predict(cv_model_06, AOSI.testing)
  test.confusionMatrix06 <- confusionMatrix(
    data = relevel(pred_class06.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc06.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 06")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix06[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix06[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs06.proc.test, rocCurve06.proc.test, auc06.test, pred_class06.test, test.confusionMatrix06)}
# calculate and extract model performance on training set (AOSI.training)
{ probs06.proc.train <- predict(cv_model_06, newdata=AOSI.training, type="prob")
  rocCurve06.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs06.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc06.train <- auc(rocCurve06.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class06.train <- predict(cv_model_06, AOSI.training)
  train.confusionMatrix06 <- confusionMatrix(
    data = relevel(pred_class06.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc06.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 06")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix06[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix06[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs06.proc.train, rocCurve06.proc.train, auc06.train, pred_class06.train, train.confusionMatrix06)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent06.proc <- predict(cv_model_06, newdata=AOSI.independent, type="prob")
  rocCurve06.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent06.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc06.independent <- auc(rocCurve06.proc)
  
  pred_class06 <- predict(cv_model_06, AOSI.independent)
  independent.confusionMatrix06 <- confusionMatrix(
    data = relevel(pred_class06, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc06.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 06")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix06[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix06[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent06.proc, rocCurve06.proc, auc06.independent, pred_class06, independent.confusionMatrix06)}

# MODEL 07 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs07.proc.test <- predict(cv_model_07, newdata=AOSI.testing, type="prob")
  rocCurve07.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs07.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc07.test <- auc(rocCurve07.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class07.test <- predict(cv_model_07, AOSI.testing)
  test.confusionMatrix07 <- confusionMatrix(
    data = relevel(pred_class07.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc07.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 07")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix07[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix07[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs07.proc.test, rocCurve07.proc.test, auc07.test, pred_class07.test, test.confusionMatrix07)}
# calculate and extract model performance on training set (AOSI.training)
{ probs07.proc.train <- predict(cv_model_07, newdata=AOSI.training, type="prob")
  rocCurve07.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs07.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc07.train <- auc(rocCurve07.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class07.train <- predict(cv_model_07, AOSI.training)
  train.confusionMatrix07 <- confusionMatrix(
    data = relevel(pred_class07.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc07.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 07")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix07[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix07[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs07.proc.train, rocCurve07.proc.train, auc07.train, pred_class07.train, train.confusionMatrix07)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent07.proc <- predict(cv_model_07, newdata=AOSI.independent, type="prob")
  rocCurve07.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent07.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc07.independent <- auc(rocCurve07.proc)
  
  pred_class07 <- predict(cv_model_07, AOSI.independent)
  independent.confusionMatrix07 <- confusionMatrix(
    data = relevel(pred_class07, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc07.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 07")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix07[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix07[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent07.proc, rocCurve07.proc, auc07.independent, pred_class07, independent.confusionMatrix07)}


# MODEL 08 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs08.proc.test <- predict(cv_model_08, newdata=AOSI.testing, type="prob")
  rocCurve08.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs08.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc08.test <- auc(rocCurve08.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class08.test <- predict(cv_model_08, AOSI.testing)
  test.confusionMatrix08 <- confusionMatrix(
    data = relevel(pred_class08.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc08.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 08")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix08[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix08[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs08.proc.test, rocCurve08.proc.test, auc08.test, pred_class08.test, test.confusionMatrix08)}
# calculate and extract model performance on training set (AOSI.training)
{ probs08.proc.train <- predict(cv_model_08, newdata=AOSI.training, type="prob")
  rocCurve08.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs08.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc08.train <- auc(rocCurve08.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class08.train <- predict(cv_model_08, AOSI.training)
  train.confusionMatrix08 <- confusionMatrix(
    data = relevel(pred_class08.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc08.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 08")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix08[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix08[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs08.proc.train, rocCurve08.proc.train, auc08.train, pred_class08.train, train.confusionMatrix08)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent08.proc <- predict(cv_model_08, newdata=AOSI.independent, type="prob")
  rocCurve08.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent08.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc08.independent <- auc(rocCurve08.proc)
  
  pred_class08 <- predict(cv_model_08, AOSI.independent)
  independent.confusionMatrix08 <- confusionMatrix(
    data = relevel(pred_class08, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc08.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 08")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix08[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix08[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent08.proc, rocCurve08.proc, auc08.independent, pred_class08, independent.confusionMatrix08)}


# MODEL 09 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs09.proc.test <- predict(cv_model_09, newdata=AOSI.testing, type="prob")
  rocCurve09.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs09.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc09.test <- auc(rocCurve09.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class09.test <- predict(cv_model_09, AOSI.testing)
  test.confusionMatrix09 <- confusionMatrix(
    data = relevel(pred_class09.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc09.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 09")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix09[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix09[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs09.proc.test, rocCurve09.proc.test, auc09.test, pred_class09.test, test.confusionMatrix09)}
# calculate and extract model performance on training set (AOSI.training)
{ probs09.proc.train <- predict(cv_model_09, newdata=AOSI.training, type="prob")
  rocCurve09.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs09.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc09.train <- auc(rocCurve09.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class09.train <- predict(cv_model_09, AOSI.training)
  train.confusionMatrix09 <- confusionMatrix(
    data = relevel(pred_class09.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc09.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 09")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix09[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix09[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs09.proc.train, rocCurve09.proc.train, auc09.train, pred_class09.train, train.confusionMatrix09)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent09.proc <- predict(cv_model_09, newdata=AOSI.independent, type="prob")
  rocCurve09.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent09.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc09.independent <- auc(rocCurve09.proc)
  
  pred_class09 <- predict(cv_model_09, AOSI.independent)
  independent.confusionMatrix09 <- confusionMatrix(
    data = relevel(pred_class09, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc09.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 09")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix09[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix09[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent09.proc, rocCurve09.proc, auc09.independent, pred_class09, independent.confusionMatrix09)}


# MODEL 10 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs10.proc.test <- predict(cv_model_10, newdata=AOSI.testing, type="prob")
  rocCurve10.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs10.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc10.test <- auc(rocCurve10.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class10.test <- predict(cv_model_10, AOSI.testing)
  test.confusionMatrix10 <- confusionMatrix(
    data = relevel(pred_class10.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc10.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 10")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix10[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix10[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs10.proc.test, rocCurve10.proc.test, auc10.test, pred_class10.test, test.confusionMatrix10)}
# calculate and extract model performance on training set (AOSI.training)
{ probs10.proc.train <- predict(cv_model_10, newdata=AOSI.training, type="prob")
  rocCurve10.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs10.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc10.train <- auc(rocCurve10.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class10.train <- predict(cv_model_10, AOSI.training)
  train.confusionMatrix10 <- confusionMatrix(
    data = relevel(pred_class10.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc10.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 10")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix10[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix10[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs10.proc.train, rocCurve10.proc.train, auc10.train, pred_class10.train, train.confusionMatrix10)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent10.proc <- predict(cv_model_10, newdata=AOSI.independent, type="prob")
  rocCurve10.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent10.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc10.independent <- auc(rocCurve10.proc)
  
  pred_class10 <- predict(cv_model_10, AOSI.independent)
  independent.confusionMatrix10 <- confusionMatrix(
    data = relevel(pred_class10, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc10.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 10")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix10[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix10[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent10.proc, rocCurve10.proc, auc10.independent, pred_class10, independent.confusionMatrix10)}


# MODEL 11 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs11.proc.test <- predict(cv_model_11, newdata=AOSI.testing, type="prob")
  rocCurve11.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs11.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc11.test <- auc(rocCurve11.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class11.test <- predict(cv_model_11, AOSI.testing)
  test.confusionMatrix11 <- confusionMatrix(
    data = relevel(pred_class11.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc11.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 11")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix11[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix11[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs11.proc.test, rocCurve11.proc.test, auc11.test, pred_class11.test, test.confusionMatrix11)}
# calculate and extract model performance on training set (AOSI.training)
{ probs11.proc.train <- predict(cv_model_11, newdata=AOSI.training, type="prob")
  rocCurve11.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs11.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc11.train <- auc(rocCurve11.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class11.train <- predict(cv_model_11, AOSI.training)
  train.confusionMatrix11 <- confusionMatrix(
    data = relevel(pred_class11.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc11.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 11")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix11[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix11[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs11.proc.train, rocCurve11.proc.train, auc11.train, pred_class11.train, train.confusionMatrix11)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent11.proc <- predict(cv_model_11, newdata=AOSI.independent, type="prob")
  rocCurve11.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent11.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc11.independent <- auc(rocCurve11.proc)
  
  pred_class11 <- predict(cv_model_11, AOSI.independent)
  independent.confusionMatrix11 <- confusionMatrix(
    data = relevel(pred_class11, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc11.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 11")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix11[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix11[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent11.proc, rocCurve11.proc, auc11.independent, pred_class11, independent.confusionMatrix11)}


# MODEL 12 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs12.proc.test <- predict(cv_model_12, newdata=AOSI.testing, type="prob")
  rocCurve12.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs12.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc12.test <- auc(rocCurve12.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class12.test <- predict(cv_model_12, AOSI.testing)
  test.confusionMatrix12 <- confusionMatrix(
    data = relevel(pred_class12.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc12.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 12")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix12[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix12[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs12.proc.test, rocCurve12.proc.test, auc12.test, pred_class12.test, test.confusionMatrix12)}
# calculate and extract model performance on training set (AOSI.training)
{ probs12.proc.train <- predict(cv_model_12, newdata=AOSI.training, type="prob")
  rocCurve12.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs12.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc12.train <- auc(rocCurve12.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class12.train <- predict(cv_model_12, AOSI.training)
  train.confusionMatrix12 <- confusionMatrix(
    data = relevel(pred_class12.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc12.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 12")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix12[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix12[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs12.proc.train, rocCurve12.proc.train, auc12.train, pred_class12.train, train.confusionMatrix12)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent12.proc <- predict(cv_model_12, newdata=AOSI.independent, type="prob")
  rocCurve12.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent12.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc12.independent <- auc(rocCurve12.proc)
  
  pred_class12 <- predict(cv_model_12, AOSI.independent)
  independent.confusionMatrix12 <- confusionMatrix(
    data = relevel(pred_class12, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc12.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 12")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix12[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix12[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent12.proc, rocCurve12.proc, auc12.independent, pred_class12, independent.confusionMatrix12)}


# MODEL 13 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs13.proc.test <- predict(cv_model_13, newdata=AOSI.testing, type="prob")
  rocCurve13.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs13.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc13.test <- auc(rocCurve13.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class13.test <- predict(cv_model_13, AOSI.testing)
  test.confusionMatrix13 <- confusionMatrix(
    data = relevel(pred_class13.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc13.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 13")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix13[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix13[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs13.proc.test, rocCurve13.proc.test, auc13.test, pred_class13.test, test.confusionMatrix13)}
# calculate and extract model performance on training set (AOSI.training)
{ probs13.proc.train <- predict(cv_model_13, newdata=AOSI.training, type="prob")
  rocCurve13.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs13.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc13.train <- auc(rocCurve13.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class13.train <- predict(cv_model_13, AOSI.training)
  train.confusionMatrix13 <- confusionMatrix(
    data = relevel(pred_class13.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc13.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 13")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix13[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix13[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs13.proc.train, rocCurve13.proc.train, auc13.train, pred_class13.train, train.confusionMatrix13)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent13.proc <- predict(cv_model_13, newdata=AOSI.independent, type="prob")
  rocCurve13.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent13.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc13.independent <- auc(rocCurve13.proc)
  
  pred_class13 <- predict(cv_model_13, AOSI.independent)
  independent.confusionMatrix13 <- confusionMatrix(
    data = relevel(pred_class13, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc13.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 13")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix13[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix13[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent13.proc, rocCurve13.proc, auc13.independent, pred_class13, independent.confusionMatrix13)}


# MODEL 14 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs14.proc.test <- predict(cv_model_14, newdata=AOSI.testing, type="prob")
  rocCurve14.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs14.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc14.test <- auc(rocCurve14.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class14.test <- predict(cv_model_14, AOSI.testing)
  test.confusionMatrix14 <- confusionMatrix(
    data = relevel(pred_class14.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc14.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 14")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix14[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix14[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs14.proc.test, rocCurve14.proc.test, auc14.test, pred_class14.test, test.confusionMatrix14)}
# calculate and extract model performance on training set (AOSI.training)
{ probs14.proc.train <- predict(cv_model_14, newdata=AOSI.training, type="prob")
  rocCurve14.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs14.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc14.train <- auc(rocCurve14.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class14.train <- predict(cv_model_14, AOSI.training)
  train.confusionMatrix14 <- confusionMatrix(
    data = relevel(pred_class14.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc14.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 14")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix14[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix14[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs14.proc.train, rocCurve14.proc.train, auc14.train, pred_class14.train, train.confusionMatrix14)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent14.proc <- predict(cv_model_14, newdata=AOSI.independent, type="prob")
  rocCurve14.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent14.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc14.independent <- auc(rocCurve14.proc)
  
  pred_class14 <- predict(cv_model_14, AOSI.independent)
  independent.confusionMatrix14 <- confusionMatrix(
    data = relevel(pred_class14, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc14.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 14")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix14[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix14[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent14.proc, rocCurve14.proc, auc14.independent, pred_class14, independent.confusionMatrix14)}


# MODEL 15 ROC CALCULATIONS 
# calculate and extract model performance on test set (AOSI.testing)
{ probs15.proc.test <- predict(cv_model_15, newdata=AOSI.testing, type="prob")
  rocCurve15.proc.test <- roc(response = AOSI.testing$dx36,
                              predictor = probs15.proc.test[,"ASD"],
                              levels = rev(levels(AOSI.testing$dx36)))
  auc15.test <- auc(rocCurve15.proc.test)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class15.test <- predict(cv_model_15, AOSI.testing)
  test.confusionMatrix15 <- confusionMatrix(
    data = relevel(pred_class15.test, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc15.test)
  number.vector.test <- append(number.vector.test, "Test - RegLogistic - Model 15")
  accuracy.vector.test  <- append(accuracy.vector.test, test.confusionMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, test.confusionMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, test.confusionMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, test.confusionMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, test.confusionMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, test.confusionMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, test.confusionMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, test.confusionMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, test.confusionMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, test.confusionMatrix15[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,test.confusionMatrix15[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs15.proc.test, rocCurve15.proc.test, auc15.test, pred_class15.test, test.confusionMatrix15)}
# calculate and extract model performance on training set (AOSI.training)
{ probs15.proc.train <- predict(cv_model_15, newdata=AOSI.training, type="prob")
  rocCurve15.proc.train <- roc(response = AOSI.training$dx36,
                               predictor = probs15.proc.train[,"ASD"],
                               levels = rev(levels(AOSI.training$dx36)))
  auc15.train <- auc(rocCurve15.proc.train)
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class15.train <- predict(cv_model_15, AOSI.training)
  train.confusionMatrix15 <- confusionMatrix(
    data = relevel(pred_class15.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")) 
  
  auc.vector.train <- append(auc.vector.train, auc15.train)
  number.vector.train <- append(number.vector.train, "Train - RegLogistic - Model 15")
  accuracy.vector.train  <- append(accuracy.vector.train, train.confusionMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, train.confusionMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, train.confusionMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, train.confusionMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, train.confusionMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, train.confusionMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, train.confusionMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, train.confusionMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, train.confusionMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, train.confusionMatrix15[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,train.confusionMatrix15[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probs15.proc.train, rocCurve15.proc.train, auc15.train, pred_class15.train, train.confusionMatrix15)}
# calculate and extract model performance on the independent testing set (AOSI.independent)
{ probsIndependent15.proc <- predict(cv_model_15, newdata=AOSI.independent, type="prob")
  rocCurve15.proc <- roc(response = AOSI.independent$dx36,
                         predictor = probsIndependent15.proc[,"ASD"],
                         levels = rev(levels(AOSI.independent$dx36)))
  auc15.independent <- auc(rocCurve15.proc)
  
  pred_class15 <- predict(cv_model_15, AOSI.independent)
  independent.confusionMatrix15 <- confusionMatrix(
    data = relevel(pred_class15, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  auc.vector.independent <- append(auc.vector.independent, auc15.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - RegLogistic - Model 15")
  accuracy.vector.independent  <- append(accuracy.vector.independent, independent.confusionMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, independent.confusionMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, independent.confusionMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, independent.confusionMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, independent.confusionMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, independent.confusionMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, independent.confusionMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, independent.confusionMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, independent.confusionMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, independent.confusionMatrix15[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, independent.confusionMatrix15[["byClass"]][["Recall"]])
  
  # removing intermediary variables to keep R's global environment clean 
  rm(probsIndependent15.proc, rocCurve15.proc, auc15.independent, pred_class15, independent.confusionMatrix15)}

# SECTION §6.0 -- generating data frames with all results together and saving finalized models ----
extracted.test.data <- data.frame(number.vector.test, accuracy.vector.test, accuracy.99CI.lower.vector.test, accuracy.99CI.upper.vector.test, Kappa.vector.test, McnemarP.vector.test, auc.vector.test, sensitivity.vector.test, specificity.vector.test, PPV.vector.test, NPV.vector.test, precision.vector.test, recall.vector.test)
extracted.test.data
write.table(extracted.test.data, file="Combined Results - RegLogistic TL10 - WITHOUT gender on TEST data", sep=",")

extracted.train.data <- data.frame(number.vector.train, accuracy.vector.train, accuracy.99CI.lower.vector.train, accuracy.99CI.upper.vector.train, Kappa.vector.train, McnemarP.vector.train, auc.vector.train, sensitivity.vector.train, specificity.vector.train, PPV.vector.train, NPV.vector.train, precision.vector.train, recall.vector.train)
extracted.train.data
write.table(extracted.train.data, file="Combined Results - RegLogistic TL10 - WITHOUT gender on TRAIN data", sep=",")

extracted.independent.data <- data.frame(number.vector.independent, accuracy.vector.independent, accuracy.99CI.lower.vector.independent, accuracy.99CI.upper.vector.independent, Kappa.vector.independent, McnemarP.vector.independent, auc.vector.independent, sensitivity.vector.independent, specificity.vector.independent, PPV.vector.independent, NPV.vector.independent, precision.vector.independent, recall.vector.independent)
extracted.independent.data
write.table(extracted.independent.data, file="Combined Results - RegLogistic TL10 - WITHOUT gender on INDEPENDENT data", sep=",")

