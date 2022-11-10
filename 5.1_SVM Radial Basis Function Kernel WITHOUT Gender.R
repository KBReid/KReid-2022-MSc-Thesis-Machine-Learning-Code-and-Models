# Support Vector Machines using radial basis function kernels *WITHOUT* Gender as a predictor variables

# SECTION §1.0 -- pre-requisite libraries/packages, importing data into R ----

# required libraries 
library(caret)      # for machine learning modelling/workflow
library(kernlab)    # for Kernel-based machine learning methods for classification
library(ROCR)       # for ROC curve analysis 
library(pROC)       # for ROC curve analysis
library(readr)      # for importing data from a .csv file into R
library(ggplot2)    # for exporting finalized .rda models for future use

# SECTION §2.0 -- code used to import the AOSI Training, Testing, and Independent data into R ----

# importing training, testing, and independent validation data into R
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
# report of model performance. It was sourced from a comment on stackoverflow; 
# https://stackoverflow.com/questions/52691761/additional-metrics-in-caret-ppv-sensitivity-specificity 

Custom.MySummary  <- function(data, lev = NULL, model = NULL){
  a1 <- defaultSummary(data, lev, model)
  b1 <- twoClassSummary(data, lev, model)  
  c1 <- prSummary(data, lev, model)
  out <- c(a1, b1, c1)
  out}

# this code takes the customized summary function and puts it into a custom trControl function we can call when training the different predictive models 
Custom.Ctrl <- trainControl(method = "cv",
                            number = 10, 
                            savePredictions = TRUE,
                            summaryFunction = Custom.MySummary,    # calls the custom summary function defined above
                            classProbs = TRUE)     

# SECTION §4.0 -- generating SVMs with radial kernels ----

# MODEL 1: Item-level data
set.seed(17274) # for reproducibility 
cv_model_01 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18,  # outcome (dx36) and the predictor variables AQ1, AQ2, etc.
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 2: Item-level data + AOSI Total Score 
set.seed(17274) # for reproducibility 
cv_model_02 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 3: Item-level data + MSEL subscales
set.seed(17274) # for reproducibility 
cv_model_03 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 4: Item-level data + Total Score + MSEL subscales
set.seed(17274) # for reproducibility 
cv_model_04 <- train(
  dx36 ~ AQ1 + AQ2 + AQ3 + AQ4 + AQ5 + AQ6 + AQ7 + AQ8 + AQ9 + AQ10 + AQ11 + AQ14 + AQ15 + AQ16 + AQ17 + AQ18 + AQTS + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 5: Total Score + MSEL subscales 
set.seed(17274) # for reproducibility 
cv_model_05 <- train(
  dx36 ~ AQTS + MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 6: MSEL subscales
set.seed(17274) # for reproducibility 
cv_model_06 <- train(
  dx36 ~ MSEL_ELCss + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 7: Total score 
set.seed(17274) # for reproducibility 
cv_model_07 <- train(
  dx36 ~ AQTS, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 8: Factor analysis items 
set.seed(17274) # for reproducibility 
cv_model_08 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 9: Factor analysis items + Total Score
set.seed(17274) # for reproducibility 
cv_model_09 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 10: Factor analysis items + MSEL subscales 
set.seed(17274) # for reproducibility 
cv_model_10 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 11: Factor analysis items + Total Score + MSEL subscales
set.seed(17274) # for reproducibility 
cv_model_11 <- train(
  dx36 ~ AQ6 + AQ8 + AQ14 + AQ16 + AQ18 + AQTS + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 12: Factor  analysis items surviving post hoc comparisons
set.seed(17274) # for reproducibility 
cv_model_12 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 13: Factor  analysis items surviving post hoc comparisons + Total Score
set.seed(17274) # for reproducibility 
cv_model_13 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + AQTS, 
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 14: Factor  analysis items surviving post hoc comparisons + MSEL subscales 
set.seed(17274) # for reproducibility 
cv_model_14 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss,
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

# MODEL 15: Factor analysis items surviving post hoc comparisons + Total SCore + MSEL subscales 
set.seed(17274) # for reproducibility 
cv_model_15 <- train(
  dx36 ~ AQ8 + AQ14 + AQ18 + AQTS + MSEL_ELCss + MSEL_VRss + MSEL_FMss + MSEL_RLss + MSEL_Elss,
  data = AOSI.training, 
  method = "svmRadial",
  metric = "ROC",
  trControl = Custom.Ctrl,
  preProcess = c("center","scale"),
  tuneLength = 10
)

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

# Generating vectors to store model statistics (when the model is applied to *INDEPENDENT* validation set data) to generate a data table for ease of recording/interpreting results
{ auc.vector.independent <- c("AUC")
  number.vector.independent <- c("Model#")
  threshold.vector.independent <- c("Threshold")
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
  recall.vector.independent <- c("Recall")
}

# Model 01 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs01.test <- predict(cv_model_01, newdata=AOSI.testing, type="prob")
  rocCurve01.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs01.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc01.test <- auc(rocCurve01.test)
  plot(rocCurve01.test, print.thres = "best")
  
  # please note that the code below is an alternate way of calculating AUC. The only
  # difference is that the code below uses the R library (ROCR) instead of the R 
  # library (pROC) used in the code above. Both approaches end up calculating the 
  # same AUC values. The pROC code was selected in this application because it is 
  # slightly shorter and uses less intermediary variables. 
  
  #prob01.test <- predict(cv_model_01, newdata=AOSI.testing, type="prob") [,2] 
  #pred01.test <- prediction(prob01.test, AOSI.testing$dx36)
  #perf01.test <- performance(pred01.test, measure = "tpr", x.measure = "fpr")
  #plot(perf01.test)
  #auc01.test <- performance(pred01.test, measure = "auc")
  #auc01.test <- auc01.test@y.values[[1]]
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class01 <- predict(cv_model_01, AOSI.testing)
  confusionMatrix01.test <- confusionMatrix(
    data = relevel(pred_class01, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc01.test)
  number.vector.test <- append(number.vector.test, "Test - Model 01 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix01.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix01.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix01.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix01.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix01.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix01.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix01.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix01.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix01.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix01.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix01.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs01.test, rocCurve01.test, auc01.test, pred_class01, confusionMatrix01.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob01.train <- predict(cv_model_01, newdata=AOSI.training, type="prob")
  rocCurve01.train <- roc(response = AOSI.training$dx36,
                          predictor = prob01.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc01.train <- auc(rocCurve01.train)
  plot(rocCurve01.train, print.thres = "best")
  
  # please note that the code below is an alternate way of calculating AUC. The only
  # difference is that the code below uses the R library (ROCR) instead of the R 
  # library (pROC) used in the code above. Both approaches end up calculating the 
  # same AUC values. The pROC code was selected in this application because it is 
  # slightly shorter and uses less intermediary variables. 
  
  #prob01.train <- predict(cv_model_01, newdata=AOSI.testing, type="prob") [,2] 
  #pred01.train <- prediction(prob01.train, AOSI.testing$dx36)
  #perf01.train <- performance(pred01.train, measure = "tpr", x.measure = "fpr")
  #plot(perf01.train)
  #auc01.train <- performance(pred01.train, measure = "auc")
  #auc01.train <- auc01.train@y.values[[1]]
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class01.train <- predict(cv_model_01, AOSI.training)
  confusionMatrix01.train <- confusionMatrix(
    data = relevel(pred_class01.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc01.train)
  number.vector.train <- append(number.vector.train, "Train - Model 01 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix01.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix01.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix01.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix01.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix01.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix01.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix01.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix01.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix01.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix01.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix01.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob01.train, rocCurve01.train, auc01.train, pred_class01.train, confusionMatrix01.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob01.independent <- predict(cv_model_01, newdata=AOSI.independent, type="prob")
  rocCurve01.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob01.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc01.independent <- auc(rocCurve01.independent)
  plot(rocCurve01.independent, print.thres = "best")
  
  # please note that the code below is an alternate way of calculating AUC. The only
  # difference is that the code below uses the R library (ROCR) instead of the R 
  # library (pROC) used in the code above. Both approaches end up calculating the 
  # same AUC values. The pROC code was selected in this application because it is 
  # slightly shorter and uses less intermediary variables. 
  
  #prob01.independent <- predict(cv_model_01, newdata=AOSI.testing, type="prob") [,2] 
  #pred01.independent <- prediction(prob01.independent, AOSI.testing$dx36)
  #perf01.independent <- performance(pred01.independent, measure = "tpr", x.measure = "fpr")
  #plot(perf01.independent)
  #auc01.independent <- performance(pred01.independent, measure = "auc")
  #auc01.independent <- auc01.independent@y.values[[1]]
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class01.independent <- predict(cv_model_01, AOSI.independent)
  confusionMatrix01.independent <- confusionMatrix(
    data = relevel(pred_class01.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc01.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 01 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix01.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix01.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix01.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix01.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix01.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix01.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix01.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix01.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix01.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix01.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix01.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob01.independent, rocCurve01.independent, auc01.independent, pred_class01.independent, confusionMatrix01.independent)
}


# Model 02 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs02.test <- predict(cv_model_02, newdata=AOSI.testing, type="prob")
  rocCurve02.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs02.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc02.test <- auc(rocCurve02.test)
  plot(rocCurve02.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class02 <- predict(cv_model_02, AOSI.testing)
  confusionMatrix02.test <- confusionMatrix(
    data = relevel(pred_class02, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc02.test)
  number.vector.test <- append(number.vector.test, "Test - Model 02 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix02.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix02.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix02.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix02.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix02.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix02.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix02.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix02.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix02.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix02.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix02.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs02.test, rocCurve02.test, auc02.test, pred_class02, confusionMatrix02.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob02.train <- predict(cv_model_02, newdata=AOSI.training, type="prob")
  rocCurve02.train <- roc(response = AOSI.training$dx36,
                          predictor = prob02.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc02.train <- auc(rocCurve02.train)
  plot(rocCurve02.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class02.train <- predict(cv_model_02, AOSI.training)
  confusionMatrix02.train <- confusionMatrix(
    data = relevel(pred_class02.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc02.train)
  number.vector.train <- append(number.vector.train, "Train - Model 02 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix02.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix02.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix02.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix02.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix02.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix02.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix02.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix02.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix02.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix02.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix02.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob02.train, rocCurve02.train, auc02.train, pred_class02.train, confusionMatrix02.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob02.independent <- predict(cv_model_02, newdata=AOSI.independent, type="prob")
  rocCurve02.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob02.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc02.independent <- auc(rocCurve02.independent)
  plot(rocCurve02.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class02.independent <- predict(cv_model_02, AOSI.independent)
  confusionMatrix02.independent <- confusionMatrix(
    data = relevel(pred_class02.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc02.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 02 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix02.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix02.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix02.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix02.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix02.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix02.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix02.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix02.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix02.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix02.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix02.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob02.independent, rocCurve02.independent, auc02.independent, pred_class02.independent, confusionMatrix02.independent)
}


# Model 03 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs03.test <- predict(cv_model_03, newdata=AOSI.testing, type="prob")
  rocCurve03.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs03.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc03.test <- auc(rocCurve03.test)
  plot(rocCurve03.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class03 <- predict(cv_model_03, AOSI.testing)
  confusionMatrix03.test <- confusionMatrix(
    data = relevel(pred_class03, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc03.test)
  number.vector.test <- append(number.vector.test, "Test - Model 03 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix03.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix03.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix03.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix03.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix03.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix03.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix03.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix03.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix03.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix03.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix03.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs03.test, rocCurve03.test, auc03.test, pred_class03, confusionMatrix03.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob03.train <- predict(cv_model_03, newdata=AOSI.training, type="prob")
  rocCurve03.train <- roc(response = AOSI.training$dx36,
                          predictor = prob03.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc03.train <- auc(rocCurve03.train)
  plot(rocCurve03.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class03.train <- predict(cv_model_03, AOSI.training)
  confusionMatrix03.train <- confusionMatrix(
    data = relevel(pred_class03.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc03.train)
  number.vector.train <- append(number.vector.train, "Train - Model 03 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix03.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix03.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix03.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix03.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix03.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix03.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix03.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix03.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix03.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix03.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix03.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob03.train, rocCurve03.train, auc03.train, pred_class03.train, confusionMatrix03.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob03.independent <- predict(cv_model_03, newdata=AOSI.independent, type="prob")
  rocCurve03.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob03.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc03.independent <- auc(rocCurve03.independent)
  plot(rocCurve03.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class03.independent <- predict(cv_model_03, AOSI.independent)
  confusionMatrix03.independent <- confusionMatrix(
    data = relevel(pred_class03.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc03.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 03 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix03.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix03.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix03.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix03.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix03.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix03.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix03.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix03.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix03.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix03.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix03.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob03.independent, rocCurve03.independent, auc03.independent, pred_class03.independent, confusionMatrix03.independent)
}


# Model 04 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs04.test <- predict(cv_model_04, newdata=AOSI.testing, type="prob")
  rocCurve04.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs04.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc04.test <- auc(rocCurve04.test)
  plot(rocCurve04.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class04 <- predict(cv_model_04, AOSI.testing)
  confusionMatrix04.test <- confusionMatrix(
    data = relevel(pred_class04, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc04.test)
  number.vector.test <- append(number.vector.test, "Test - Model 04 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix04.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix04.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix04.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix04.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix04.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix04.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix04.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix04.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix04.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix04.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix04.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs04.test, rocCurve04.test, auc04.test, pred_class04, confusionMatrix04.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob04.train <- predict(cv_model_04, newdata=AOSI.training, type="prob")
  rocCurve04.train <- roc(response = AOSI.training$dx36,
                          predictor = prob04.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc04.train <- auc(rocCurve04.train)
  plot(rocCurve04.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class04.train <- predict(cv_model_04, AOSI.training)
  confusionMatrix04.train <- confusionMatrix(
    data = relevel(pred_class04.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc04.train)
  number.vector.train <- append(number.vector.train, "Train - Model 04 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix04.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix04.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix04.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix04.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix04.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix04.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix04.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix04.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix04.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix04.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix04.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob04.train, rocCurve04.train, auc04.train, pred_class04.train, confusionMatrix04.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob04.independent <- predict(cv_model_04, newdata=AOSI.independent, type="prob")
  rocCurve04.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob04.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc04.independent <- auc(rocCurve04.independent)
  plot(rocCurve04.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class04.independent <- predict(cv_model_04, AOSI.independent)
  confusionMatrix04.independent <- confusionMatrix(
    data = relevel(pred_class04.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc04.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 04 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix04.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix04.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix04.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix04.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix04.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix04.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix04.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix04.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix04.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix04.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix04.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob04.independent, rocCurve04.independent, auc04.independent, pred_class04.independent, confusionMatrix04.independent)
}


# Model 05 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs05.test <- predict(cv_model_05, newdata=AOSI.testing, type="prob")
  rocCurve05.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs05.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc05.test <- auc(rocCurve05.test)
  plot(rocCurve05.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class05 <- predict(cv_model_05, AOSI.testing)
  confusionMatrix05.test <- confusionMatrix(
    data = relevel(pred_class05, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc05.test)
  number.vector.test <- append(number.vector.test, "Test - Model 05 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix05.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix05.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix05.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix05.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix05.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix05.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix05.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix05.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix05.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix05.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix05.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs05.test, rocCurve05.test, auc05.test, pred_class05, confusionMatrix05.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob05.train <- predict(cv_model_05, newdata=AOSI.training, type="prob")
  rocCurve05.train <- roc(response = AOSI.training$dx36,
                          predictor = prob05.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc05.train <- auc(rocCurve05.train)
  plot(rocCurve05.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class05.train <- predict(cv_model_05, AOSI.training)
  confusionMatrix05.train <- confusionMatrix(
    data = relevel(pred_class05.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc05.train)
  number.vector.train <- append(number.vector.train, "Train - Model 05 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix05.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix05.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix05.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix05.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix05.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix05.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix05.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix05.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix05.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix05.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix05.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob05.train, rocCurve05.train, auc05.train, pred_class05.train, confusionMatrix05.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob05.independent <- predict(cv_model_05, newdata=AOSI.independent, type="prob")
  rocCurve05.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob05.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc05.independent <- auc(rocCurve05.independent)
  plot(rocCurve05.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class05.independent <- predict(cv_model_05, AOSI.independent)
  confusionMatrix05.independent <- confusionMatrix(
    data = relevel(pred_class05.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc05.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 05 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix05.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix05.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix05.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix05.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix05.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix05.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix05.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix05.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix05.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix05.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix05.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob05.independent, rocCurve05.independent, auc05.independent, pred_class05.independent, confusionMatrix05.independent)
}


# Model 06 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs06.test <- predict(cv_model_06, newdata=AOSI.testing, type="prob")
  rocCurve06.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs06.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc06.test <- auc(rocCurve06.test)
  plot(rocCurve06.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class06 <- predict(cv_model_06, AOSI.testing)
  confusionMatrix06.test <- confusionMatrix(
    data = relevel(pred_class06, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc06.test)
  number.vector.test <- append(number.vector.test, "Test - Model 06 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix06.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix06.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix06.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix06.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix06.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix06.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix06.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix06.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix06.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix06.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix06.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs06.test, rocCurve06.test, auc06.test, pred_class06, confusionMatrix06.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob06.train <- predict(cv_model_06, newdata=AOSI.training, type="prob")
  rocCurve06.train <- roc(response = AOSI.training$dx36,
                          predictor = prob06.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc06.train <- auc(rocCurve06.train)
  plot(rocCurve06.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class06.train <- predict(cv_model_06, AOSI.training)
  confusionMatrix06.train <- confusionMatrix(
    data = relevel(pred_class06.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc06.train)
  number.vector.train <- append(number.vector.train, "Train - Model 06 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix06.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix06.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix06.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix06.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix06.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix06.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix06.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix06.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix06.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix06.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix06.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob06.train, rocCurve06.train, auc06.train, pred_class06.train, confusionMatrix06.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob06.independent <- predict(cv_model_06, newdata=AOSI.independent, type="prob")
  rocCurve06.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob06.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc06.independent <- auc(rocCurve06.independent)
  plot(rocCurve06.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class06.independent <- predict(cv_model_06, AOSI.independent)
  confusionMatrix06.independent <- confusionMatrix(
    data = relevel(pred_class06.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc06.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 06 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix06.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix06.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix06.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix06.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix06.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix06.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix06.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix06.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix06.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix06.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix06.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob06.independent, rocCurve06.independent, auc06.independent, pred_class06.independent, confusionMatrix06.independent)
}


# Model 07 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs07.test <- predict(cv_model_07, newdata=AOSI.testing, type="prob")
  rocCurve07.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs07.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc07.test <- auc(rocCurve07.test)
  plot(rocCurve07.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class07 <- predict(cv_model_07, AOSI.testing)
  confusionMatrix07.test <- confusionMatrix(
    data = relevel(pred_class07, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc07.test)
  number.vector.test <- append(number.vector.test, "Test - Model 07 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix07.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix07.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix07.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix07.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix07.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix07.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix07.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix07.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix07.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix07.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix07.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs07.test, rocCurve07.test, auc07.test, pred_class07, confusionMatrix07.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob07.train <- predict(cv_model_07, newdata=AOSI.training, type="prob")
  rocCurve07.train <- roc(response = AOSI.training$dx36,
                          predictor = prob07.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc07.train <- auc(rocCurve07.train)
  plot(rocCurve07.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class07.train <- predict(cv_model_07, AOSI.training)
  confusionMatrix07.train <- confusionMatrix(
    data = relevel(pred_class07.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc07.train)
  number.vector.train <- append(number.vector.train, "Train - Model 07 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix07.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix07.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix07.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix07.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix07.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix07.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix07.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix07.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix07.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix07.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix07.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob07.train, rocCurve07.train, auc07.train, pred_class07.train, confusionMatrix07.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob07.independent <- predict(cv_model_07, newdata=AOSI.independent, type="prob")
  rocCurve07.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob07.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc07.independent <- auc(rocCurve07.independent)
  plot(rocCurve07.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class07.independent <- predict(cv_model_07, AOSI.independent)
  confusionMatrix07.independent <- confusionMatrix(
    data = relevel(pred_class07.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc07.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 07 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix07.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix07.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix07.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix07.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix07.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix07.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix07.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix07.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix07.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix07.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix07.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob07.independent, rocCurve07.independent, auc07.independent, pred_class07.independent, confusionMatrix07.independent)
}


# Model 08 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs08.test <- predict(cv_model_08, newdata=AOSI.testing, type="prob")
  rocCurve08.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs08.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc08.test <- auc(rocCurve08.test)
  plot(rocCurve08.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class08 <- predict(cv_model_08, AOSI.testing)
  confusionMatrix08.test <- confusionMatrix(
    data = relevel(pred_class08, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc08.test)
  number.vector.test <- append(number.vector.test, "Test - Model 08 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix08.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix08.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix08.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix08.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix08.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix08.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix08.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix08.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix08.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix08.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix08.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs08.test, rocCurve08.test, auc08.test, pred_class08, confusionMatrix08.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob08.train <- predict(cv_model_08, newdata=AOSI.training, type="prob")
  rocCurve08.train <- roc(response = AOSI.training$dx36,
                          predictor = prob08.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc08.train <- auc(rocCurve08.train)
  plot(rocCurve08.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class08.train <- predict(cv_model_08, AOSI.training)
  confusionMatrix08.train <- confusionMatrix(
    data = relevel(pred_class08.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc08.train)
  number.vector.train <- append(number.vector.train, "Train - Model 08 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix08.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix08.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix08.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix08.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix08.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix08.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix08.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix08.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix08.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix08.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix08.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob08.train, rocCurve08.train, auc08.train, pred_class08.train, confusionMatrix08.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob08.independent <- predict(cv_model_08, newdata=AOSI.independent, type="prob")
  rocCurve08.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob08.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc08.independent <- auc(rocCurve08.independent)
  plot(rocCurve08.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class08.independent <- predict(cv_model_08, AOSI.independent)
  confusionMatrix08.independent <- confusionMatrix(
    data = relevel(pred_class08.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc08.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 08 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix08.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix08.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix08.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix08.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix08.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix08.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix08.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix08.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix08.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix08.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix08.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob08.independent, rocCurve08.independent, auc08.independent, pred_class08.independent, confusionMatrix08.independent)
}


# Model 09 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs09.test <- predict(cv_model_09, newdata=AOSI.testing, type="prob")
  rocCurve09.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs09.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc09.test <- auc(rocCurve09.test)
  plot(rocCurve09.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class09 <- predict(cv_model_09, AOSI.testing)
  confusionMatrix09.test <- confusionMatrix(
    data = relevel(pred_class09, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc09.test)
  number.vector.test <- append(number.vector.test, "Test - Model 09 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix09.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix09.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix09.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix09.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix09.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix09.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix09.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix09.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix09.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix09.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix09.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs09.test, rocCurve09.test, auc09.test, pred_class09, confusionMatrix09.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob09.train <- predict(cv_model_09, newdata=AOSI.training, type="prob")
  rocCurve09.train <- roc(response = AOSI.training$dx36,
                          predictor = prob09.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc09.train <- auc(rocCurve09.train)
  plot(rocCurve09.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class09.train <- predict(cv_model_09, AOSI.training)
  confusionMatrix09.train <- confusionMatrix(
    data = relevel(pred_class09.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc09.train)
  number.vector.train <- append(number.vector.train, "Train - Model 09 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix09.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix09.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix09.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix09.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix09.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix09.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix09.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix09.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix09.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix09.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix09.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob09.train, rocCurve09.train, auc09.train, pred_class09.train, confusionMatrix09.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob09.independent <- predict(cv_model_09, newdata=AOSI.independent, type="prob")
  rocCurve09.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob09.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc09.independent <- auc(rocCurve09.independent)
  plot(rocCurve09.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class09.independent <- predict(cv_model_09, AOSI.independent)
  confusionMatrix09.independent <- confusionMatrix(
    data = relevel(pred_class09.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc09.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 09 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix09.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix09.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix09.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix09.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix09.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix09.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix09.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix09.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix09.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix09.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix09.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob09.independent, rocCurve09.independent, auc09.independent, pred_class09.independent, confusionMatrix09.independent)
}


# Model 10 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs10.test <- predict(cv_model_10, newdata=AOSI.testing, type="prob")
  rocCurve10.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs10.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc10.test <- auc(rocCurve10.test)
  plot(rocCurve10.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class10 <- predict(cv_model_10, AOSI.testing)
  confusionMatrix10.test <- confusionMatrix(
    data = relevel(pred_class10, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc10.test)
  number.vector.test <- append(number.vector.test, "Test - Model 10 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix10.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix10.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix10.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix10.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix10.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix10.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix10.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix10.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix10.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix10.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix10.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs10.test, rocCurve10.test, auc10.test, pred_class10, confusionMatrix10.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob10.train <- predict(cv_model_10, newdata=AOSI.training, type="prob")
  rocCurve10.train <- roc(response = AOSI.training$dx36,
                          predictor = prob10.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc10.train <- auc(rocCurve10.train)
  plot(rocCurve10.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class10.train <- predict(cv_model_10, AOSI.training)
  confusionMatrix10.train <- confusionMatrix(
    data = relevel(pred_class10.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc10.train)
  number.vector.train <- append(number.vector.train, "Train - Model 10 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix10.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix10.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix10.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix10.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix10.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix10.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix10.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix10.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix10.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix10.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix10.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob10.train, rocCurve10.train, auc10.train, pred_class10.train, confusionMatrix10.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob10.independent <- predict(cv_model_10, newdata=AOSI.independent, type="prob")
  rocCurve10.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob10.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc10.independent <- auc(rocCurve10.independent)
  plot(rocCurve10.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class10.independent <- predict(cv_model_10, AOSI.independent)
  confusionMatrix10.independent <- confusionMatrix(
    data = relevel(pred_class10.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc10.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 10 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix10.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix10.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix10.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix10.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix10.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix10.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix10.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix10.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix10.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix10.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix10.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob10.independent, rocCurve10.independent, auc10.independent, pred_class10.independent, confusionMatrix10.independent)
}


# Model 11 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs11.test <- predict(cv_model_11, newdata=AOSI.testing, type="prob")
  rocCurve11.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs11.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc11.test <- auc(rocCurve11.test)
  plot(rocCurve11.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class11 <- predict(cv_model_11, AOSI.testing)
  confusionMatrix11.test <- confusionMatrix(
    data = relevel(pred_class11, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc11.test)
  number.vector.test <- append(number.vector.test, "Test - Model 11 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix11.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix11.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix11.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix11.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix11.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix11.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix11.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix11.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix11.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix11.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix11.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs11.test, rocCurve11.test, auc11.test, pred_class11, confusionMatrix11.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob11.train <- predict(cv_model_11, newdata=AOSI.training, type="prob")
  rocCurve11.train <- roc(response = AOSI.training$dx36,
                          predictor = prob11.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc11.train <- auc(rocCurve11.train)
  plot(rocCurve11.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class11.train <- predict(cv_model_11, AOSI.training)
  confusionMatrix11.train <- confusionMatrix(
    data = relevel(pred_class11.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc11.train)
  number.vector.train <- append(number.vector.train, "Train - Model 11 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix11.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix11.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix11.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix11.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix11.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix11.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix11.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix11.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix11.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix11.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix11.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob11.train, rocCurve11.train, auc11.train, pred_class11.train, confusionMatrix11.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob11.independent <- predict(cv_model_11, newdata=AOSI.independent, type="prob")
  rocCurve11.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob11.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc11.independent <- auc(rocCurve11.independent)
  plot(rocCurve11.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class11.independent <- predict(cv_model_11, AOSI.independent)
  confusionMatrix11.independent <- confusionMatrix(
    data = relevel(pred_class11.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc11.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 11 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix11.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix11.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix11.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix11.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix11.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix11.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix11.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix11.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix11.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix11.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix11.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob11.independent, rocCurve11.independent, auc11.independent, pred_class11.independent, confusionMatrix11.independent)
}


# Model 12 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs12.test <- predict(cv_model_12, newdata=AOSI.testing, type="prob")
  rocCurve12.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs12.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc12.test <- auc(rocCurve12.test)
  plot(rocCurve12.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class12 <- predict(cv_model_12, AOSI.testing)
  confusionMatrix12.test <- confusionMatrix(
    data = relevel(pred_class12, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc12.test)
  number.vector.test <- append(number.vector.test, "Test - Model 12 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix12.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix12.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix12.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix12.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix12.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix12.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix12.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix12.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix12.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix12.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix12.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs12.test, rocCurve12.test, auc12.test, pred_class12, confusionMatrix12.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob12.train <- predict(cv_model_12, newdata=AOSI.training, type="prob")
  rocCurve12.train <- roc(response = AOSI.training$dx36,
                          predictor = prob12.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc12.train <- auc(rocCurve12.train)
  plot(rocCurve12.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class12.train <- predict(cv_model_12, AOSI.training)
  confusionMatrix12.train <- confusionMatrix(
    data = relevel(pred_class12.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc12.train)
  number.vector.train <- append(number.vector.train, "Train - Model 12 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix12.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix12.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix12.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix12.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix12.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix12.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix12.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix12.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix12.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix12.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix12.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob12.train, rocCurve12.train, auc12.train, pred_class12.train, confusionMatrix12.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob12.independent <- predict(cv_model_12, newdata=AOSI.independent, type="prob")
  rocCurve12.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob12.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc12.independent <- auc(rocCurve12.independent)
  plot(rocCurve12.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class12.independent <- predict(cv_model_12, AOSI.independent)
  confusionMatrix12.independent <- confusionMatrix(
    data = relevel(pred_class12.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc12.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 12 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix12.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix12.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix12.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix12.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix12.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix12.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix12.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix12.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix12.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix12.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix12.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob12.independent, rocCurve12.independent, auc12.independent, pred_class12.independent, confusionMatrix12.independent)
}


# Model 13 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs13.test <- predict(cv_model_13, newdata=AOSI.testing, type="prob")
  rocCurve13.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs13.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc13.test <- auc(rocCurve13.test)
  plot(rocCurve13.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class13 <- predict(cv_model_13, AOSI.testing)
  confusionMatrix13.test <- confusionMatrix(
    data = relevel(pred_class13, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc13.test)
  number.vector.test <- append(number.vector.test, "Test - Model 13 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix13.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix13.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix13.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix13.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix13.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix13.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix13.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix13.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix13.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix13.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix13.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs13.test, rocCurve13.test, auc13.test, pred_class13, confusionMatrix13.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob13.train <- predict(cv_model_13, newdata=AOSI.training, type="prob")
  rocCurve13.train <- roc(response = AOSI.training$dx36,
                          predictor = prob13.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc13.train <- auc(rocCurve13.train)
  plot(rocCurve13.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class13.train <- predict(cv_model_13, AOSI.training)
  confusionMatrix13.train <- confusionMatrix(
    data = relevel(pred_class13.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc13.train)
  number.vector.train <- append(number.vector.train, "Train - Model 13 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix13.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix13.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix13.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix13.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix13.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix13.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix13.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix13.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix13.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix13.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix13.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob13.train, rocCurve13.train, auc13.train, pred_class13.train, confusionMatrix13.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob13.independent <- predict(cv_model_13, newdata=AOSI.independent, type="prob")
  rocCurve13.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob13.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc13.independent <- auc(rocCurve13.independent)
  plot(rocCurve13.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class13.independent <- predict(cv_model_13, AOSI.independent)
  confusionMatrix13.independent <- confusionMatrix(
    data = relevel(pred_class13.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc13.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 13 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix13.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix13.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix13.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix13.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix13.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix13.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix13.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix13.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix13.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix13.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix13.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob13.independent, rocCurve13.independent, auc13.independent, pred_class13.independent, confusionMatrix13.independent)
}


# Model 14 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs14.test <- predict(cv_model_14, newdata=AOSI.testing, type="prob")
  rocCurve14.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs14.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc14.test <- auc(rocCurve14.test)
  plot(rocCurve14.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class14 <- predict(cv_model_14, AOSI.testing)
  confusionMatrix14.test <- confusionMatrix(
    data = relevel(pred_class14, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc14.test)
  number.vector.test <- append(number.vector.test, "Test - Model 14 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix14.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix14.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix14.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix14.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix14.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix14.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix14.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix14.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix14.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix14.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix14.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs14.test, rocCurve14.test, auc14.test, pred_class14, confusionMatrix14.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob14.train <- predict(cv_model_14, newdata=AOSI.training, type="prob")
  rocCurve14.train <- roc(response = AOSI.training$dx36,
                          predictor = prob14.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc14.train <- auc(rocCurve14.train)
  plot(rocCurve14.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class14.train <- predict(cv_model_14, AOSI.training)
  confusionMatrix14.train <- confusionMatrix(
    data = relevel(pred_class14.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc14.train)
  number.vector.train <- append(number.vector.train, "Train - Model 14 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix14.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix14.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix14.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix14.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix14.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix14.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix14.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix14.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix14.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix14.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix14.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob14.train, rocCurve14.train, auc14.train, pred_class14.train, confusionMatrix14.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob14.independent <- predict(cv_model_14, newdata=AOSI.independent, type="prob")
  rocCurve14.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob14.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc14.independent <- auc(rocCurve14.independent)
  plot(rocCurve14.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class14.independent <- predict(cv_model_14, AOSI.independent)
  confusionMatrix14.independent <- confusionMatrix(
    data = relevel(pred_class14.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc14.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 14 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix14.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix14.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix14.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix14.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix14.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix14.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix14.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix14.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix14.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix14.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix14.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob14.independent, rocCurve14.independent, auc14.independent, pred_class14.independent, confusionMatrix14.independent)
}


# Model 15 performance calculations on testing, training, and independent data
# calculate and extract model performance on the *TESTING* set.
{
  # Compute AUC for predicting Class with the model
  probs15.test <- predict(cv_model_15, newdata=AOSI.testing, type="prob")
  rocCurve15.test <- roc(response = AOSI.testing$dx36,
                         predictor = probs15.test[,"ASD"],
                         levels = rev(levels(AOSI.testing$dx36)))
  auc15.test <- auc(rocCurve15.test)
  plot(rocCurve15.test, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class15 <- predict(cv_model_15, AOSI.testing)
  confusionMatrix15.test <- confusionMatrix(
    data = relevel(pred_class15, ref = "ASD"), 
    reference = relevel(AOSI.testing$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.test <- append(auc.vector.test, auc15.test)
  number.vector.test <- append(number.vector.test, "Test - Model 15 - SVM radial WITHOUT gender")
  accuracy.vector.test  <- append(accuracy.vector.test, confusionMatrix15.test[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, confusionMatrix15.test[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, confusionMatrix15.test[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, confusionMatrix15.test[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, confusionMatrix15.test[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, confusionMatrix15.test[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, confusionMatrix15.test[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, confusionMatrix15.test[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, confusionMatrix15.test[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, confusionMatrix15.test[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,confusionMatrix15.test[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(probs15.test, rocCurve15.test, auc15.test, pred_class15, confusionMatrix15.test)
}
# calculate and extract model performance on the *TRAINING* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob15.train <- predict(cv_model_15, newdata=AOSI.training, type="prob")
  rocCurve15.train <- roc(response = AOSI.training$dx36,
                          predictor = prob15.train[,"ASD"],
                          levels = rev(levels(AOSI.training$dx36)))
  auc15.train <- auc(rocCurve15.train)
  plot(rocCurve15.train, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class15.train <- predict(cv_model_15, AOSI.training)
  confusionMatrix15.train <- confusionMatrix(
    data = relevel(pred_class15.train, ref = "ASD"), 
    reference = relevel(AOSI.training$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.train <- append(auc.vector.train, auc15.train)
  number.vector.train <- append(number.vector.train, "Train - Model 15 - SVM radial WITHOUT gender")
  accuracy.vector.train  <- append(accuracy.vector.train, confusionMatrix15.train[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, confusionMatrix15.train[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, confusionMatrix15.train[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, confusionMatrix15.train[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, confusionMatrix15.train[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, confusionMatrix15.train[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, confusionMatrix15.train[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, confusionMatrix15.train[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, confusionMatrix15.train[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, confusionMatrix15.train[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train, confusionMatrix15.train[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob15.train, rocCurve15.train, auc15.train, pred_class15.train, confusionMatrix15.train)
}
# calculate and extract model performance on the *INDEPENDENT* set. 
{ 
  # Compute AUC for predicting Class with the model
  prob15.independent <- predict(cv_model_15, newdata=AOSI.independent, type="prob")
  rocCurve15.independent <- roc(response = AOSI.independent$dx36,
                                predictor = prob15.independent[,"ASD"],
                                levels = rev(levels(AOSI.independent$dx36)))
  auc15.independent <- auc(rocCurve15.independent)
  plot(rocCurve15.independent, print.thres = "best")
  
  # compute desired statistics (accuracy, sens, spec, PPV, NPV, etc.) when applied to test set
  pred_class15.independent <- predict(cv_model_15, AOSI.independent)
  confusionMatrix15.independent <- confusionMatrix(
    data = relevel(pred_class15.independent, ref = "ASD"), 
    reference = relevel(AOSI.independent$dx36, ref = "ASD")
  ) 
  
  # extracting model performance on test set. 
  auc.vector.independent <- append(auc.vector.independent, auc15.independent)
  number.vector.independent <- append(number.vector.independent, "Independent - Model 15 - SVM radial WITHOUT gender")
  accuracy.vector.independent  <- append(accuracy.vector.independent, confusionMatrix15.independent[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, confusionMatrix15.independent[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, confusionMatrix15.independent[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, confusionMatrix15.independent[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, confusionMatrix15.independent[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, confusionMatrix15.independent[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, confusionMatrix15.independent[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, confusionMatrix15.independent[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, confusionMatrix15.independent[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, confusionMatrix15.independent[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent, confusionMatrix15.independent[["byClass"]][["Recall"]])
  
  # remove variables to keep RStudio environment clean and not cluttered
  rm(prob15.independent, rocCurve15.independent, auc15.independent, pred_class15.independent, confusionMatrix15.independent)
}

# SECTION §6.0 -- generating data frames with all results together and saving finalized models ----
extracted.train.data <- data.frame(number.vector.train, accuracy.vector.train, accuracy.99CI.lower.vector.train, accuracy.99CI.upper.vector.train, Kappa.vector.train, McnemarP.vector.train, auc.vector.train, sensitivity.vector.train, specificity.vector.train, PPV.vector.train, NPV.vector.train, precision.vector.train, recall.vector.train)
extracted.train.data
write.table(extracted.train.data, file="Combined SVM radial Results WITHOUT gender on TRAIN data.csv", sep=",")

extracted.test.data <- data.frame(number.vector.test, accuracy.vector.test, accuracy.99CI.lower.vector.test, accuracy.99CI.upper.vector.test, Kappa.vector.test, McnemarP.vector.test, auc.vector.test, sensitivity.vector.test, specificity.vector.test, PPV.vector.test, NPV.vector.test, precision.vector.test, recall.vector.test)
extracted.test.data
write.table(extracted.test.data, file="Combined SVM radial Results WITHOUT gender on TEST data.csv", sep=",")

extracted.independent.data <- data.frame(number.vector.independent, accuracy.vector.independent, accuracy.99CI.lower.vector.independent, accuracy.99CI.upper.vector.independent, Kappa.vector.independent, McnemarP.vector.independent, auc.vector.independent, sensitivity.vector.independent, specificity.vector.independent, PPV.vector.independent, NPV.vector.independent, precision.vector.independent, recall.vector.independent)
extracted.independent.data
write.table(extracted.independent.data, file="Combined SVM radial Results WITHOUT gender on INDEPENDENT data.csv", sep=",")

save(cv_model_01, file = "SVM radial TL10 WITHOUT gender cv_model_01.rda")
save(cv_model_02, file = "SVM radial TL10 WITHOUT gender cv_model_02.rda")
save(cv_model_03, file = "SVM radial TL10 WITHOUT gender cv_model_03.rda")
save(cv_model_04, file = "SVM radial TL10 WITHOUT gender cv_model_04.rda")
save(cv_model_05, file = "SVM radial TL10 WITHOUT gender cv_model_05.rda")
save(cv_model_06, file = "SVM radial TL10 WITHOUT gender cv_model_06.rda")
save(cv_model_07, file = "SVM radial TL10 WITHOUT gender cv_model_07.rda")
save(cv_model_08, file = "SVM radial TL10 WITHOUT gender cv_model_08.rda")
save(cv_model_09, file = "SVM radial TL10 WITHOUT gender cv_model_09.rda")
save(cv_model_10, file = "SVM radial TL10 WITHOUT gender cv_model_10.rda")
save(cv_model_11, file = "SVM radial TL10 WITHOUT gender cv_model_11.rda")
save(cv_model_12, file = "SVM radial TL10 WITHOUT gender cv_model_12.rda")
save(cv_model_13, file = "SVM radial TL10 WITHOUT gender cv_model_13.rda")
save(cv_model_14, file = "SVM radial TL10 WITHOUT gender cv_model_14.rda")
save(cv_model_15, file = "SVM radial TL10 WITHOUT gender cv_model_15.rda")
