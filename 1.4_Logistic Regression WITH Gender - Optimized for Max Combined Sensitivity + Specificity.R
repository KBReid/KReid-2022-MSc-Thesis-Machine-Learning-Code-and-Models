# Logistic Regression Models *WITH* Gender optimized for maximum combined sensitivity and specificity 

# DESCRIPTION ----
# Please note that this code does not generate logistic models. 
# Instead, you will need to import the different models in question into the R / 
# RStudio working environment (cv_model_01 through cv_model_15) and then run the 
# code sections in chunks. 

# The whole idea behind this is to find a logistic regression decision threshold
# (which is a value between 0 and 1) for logistic regression models generated
# using AOSI data that yields the highest possible combined sensitivity
# and specificity. 

# This work was completed because after logistic model generation, the models will, 
# by default, use a decision threshold of 0.500 which may or may not be the optimal 
# decision boundary when classifying for ASD (in the case of models built using AOSI
# data trying to predict 36 month ASD diagnoses). 

# Required R pre-requisites  ---- 
library(pROC)     # for ROC curve analysis 

# Logistic models applied to and optimized for **TRAINING** data ----

# This code generates vectors we can use to store performance statistics in.
# These vectors will all be combined into a single data frame that can be exported
# in whatever format we want. 
{ auc.vector.train <- c("AUC")
  number.vector.train <- c("Model#")
  threshold.vector.train <- c("Threshold")
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

# MODEL 1 
# This code will result in a ROC curve plot with the probability decision boundary
# that yields the highest combined model sensitivity + specificity marked on it when 
# applied to the *TRAINING* data set. 
probsTrain01 <- predict(cv_model_01, newdata=AOSI.training, type="prob")
rocCurve01 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain01[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc01 <- auc(rocCurve01)
plot(rocCurve01, print.thres = "best") 

# The code below will let us manually test different logistic regression probability 
# decision thresholds we want between 0 and 1. In our case, we want to take the 
# probability decision boundary point identified above that yields the highest 
# maximal combined sensitivity + specificity and plug it into the code below 
# for threshold01 
probsTest01 <- predict(cv_model_01, AOSI.training, type = "prob")
threshold01 <- 0.393    # corresponds to the optimum probability decision threshold
pred01      <- factor( ifelse(probsTest01[, "ASD"] > threshold01, "ASD", "N") )
pred01      <- relevel(pred01, "ASD")    
Optimal.tuningMatrix01 <- confusionMatrix(pred01, AOSI.training$dx36)

# This code extracts the performance statistics (accuracy, kappa, sensitivity,
# specificity, etc.) from model 1 when the specified decision boundary was used.  
{ auc.vector.train <- append(auc.vector.train, auc01)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 01")
  threshold.vector.train <- append(threshold.vector.train, threshold01)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix01[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix01[["byClass"]][["Recall"]])}

# MODEL 2
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain02 <- predict(cv_model_02, newdata=AOSI.training, type="prob")
rocCurve02 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain02[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc02 <- auc(rocCurve02)
plot(rocCurve02, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest02 <- predict(cv_model_02, AOSI.training, type = "prob")
threshold02 <- 0.393
pred02      <- factor( ifelse(probsTest02[, "ASD"] > threshold02, "ASD", "N") )
pred02      <- relevel(pred02, "ASD")    
Optimal.tuningMatrix02 <- confusionMatrix(pred02, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc02)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 02")
  threshold.vector.train <- append(threshold.vector.train, threshold02)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix02[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix02[["byClass"]][["Recall"]])}

# MODEL 3
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain03 <- predict(cv_model_03, newdata=AOSI.training, type="prob")
rocCurve03 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain03[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc03 <- auc(rocCurve03)
plot(rocCurve03, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest03 <- predict(cv_model_03, AOSI.training, type = "prob")
threshold03 <- 0.338
pred03      <- factor( ifelse(probsTest03[, "ASD"] > threshold03, "ASD", "N") )
pred03      <- relevel(pred03, "ASD")    
Optimal.tuningMatrix03 <- confusionMatrix(pred03, AOSI.training$dx36)


# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc03)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 03")
  threshold.vector.train <- append(threshold.vector.train, threshold03)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix03[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix03[["byClass"]][["Recall"]])}

# Model 4
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain04 <- predict(cv_model_04, newdata=AOSI.training, type="prob")
rocCurve04 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain04[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc04 <- auc(rocCurve04)
plot(rocCurve04, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest04 <- predict(cv_model_04, AOSI.training, type = "prob")
threshold04 <- 0.338
pred04      <- factor( ifelse(probsTest04[, "ASD"] > threshold04, "ASD", "N") )
pred04      <- relevel(pred04, "ASD")    
Optimal.tuningMatrix04 <- confusionMatrix(pred04, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc04)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 04")
  threshold.vector.train <- append(threshold.vector.train, threshold04)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix04[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix04[["byClass"]][["Recall"]])}

# Model 5
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain05 <- predict(cv_model_05, newdata=AOSI.training, type="prob")
rocCurve05 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain05[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc05 <- auc(rocCurve05)
plot(rocCurve05, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest05 <- predict(cv_model_05, AOSI.training, type = "prob")
threshold05 <- 0.298
pred05      <- factor( ifelse(probsTest05[, "ASD"] > threshold05, "ASD", "N") )
pred05      <- relevel(pred05, "ASD")    
Optimal.tuningMatrix05 <- confusionMatrix(pred05, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc05)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 05")
  threshold.vector.train <- append(threshold.vector.train, threshold05)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix05[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix05[["byClass"]][["Recall"]])}

# Model 6
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain06 <- predict(cv_model_06, newdata=AOSI.training, type="prob")
rocCurve06 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain06[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc06 <- auc(rocCurve06)
plot(rocCurve06, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest06 <- predict(cv_model_06, AOSI.training, type = "prob")
threshold06 <- 0.219
pred06      <- factor( ifelse(probsTest06[, "ASD"] > threshold06, "ASD", "N") )
pred06      <- relevel(pred06, "ASD")    
Optimal.tuningMatrix06 <- confusionMatrix(pred06, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc06)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 06")
  threshold.vector.train <- append(threshold.vector.train, threshold06)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix06[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix06[["byClass"]][["Recall"]])}

# Model 7
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain07 <- predict(cv_model_07, newdata=AOSI.training, type="prob")
rocCurve07 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain07[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc07 <- auc(rocCurve07)
plot(rocCurve07, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest07 <- predict(cv_model_07, AOSI.training, type = "prob")
threshold07 <- 0.330
pred07      <- factor( ifelse(probsTest07[, "ASD"] > threshold07, "ASD", "N") )
pred07      <- relevel(pred07, "ASD")    
Optimal.tuningMatrix07 <- confusionMatrix(pred07, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc07)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 07")
  threshold.vector.train <- append(threshold.vector.train, threshold07)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix07[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix07[["byClass"]][["Recall"]])}

# Model 8
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain08 <- predict(cv_model_08, newdata=AOSI.training, type="prob")
rocCurve08 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain08[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc08 <- auc(rocCurve08)
plot(rocCurve08, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest08 <- predict(cv_model_08, AOSI.training, type = "prob")
threshold08 <- 0.248
pred08      <- factor( ifelse(probsTest08[, "ASD"] > threshold08, "ASD", "N") )
pred08      <- relevel(pred08, "ASD")    
Optimal.tuningMatrix08 <- confusionMatrix(pred08, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc08)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 08")
  threshold.vector.train <- append(threshold.vector.train, threshold08)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix08[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix08[["byClass"]][["Recall"]])}

# Model 9
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain09 <- predict(cv_model_09, newdata=AOSI.training, type="prob")
rocCurve09 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain09[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc09 <- auc(rocCurve09)
plot(rocCurve09, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest09 <- predict(cv_model_09, AOSI.training, type = "prob")
threshold09 <- 0.357
pred09      <- factor( ifelse(probsTest09[, "ASD"] > threshold09, "ASD", "N") )
pred09      <- relevel(pred09, "ASD")    
Optimal.tuningMatrix09 <- confusionMatrix(pred09, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc09)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 09")
  threshold.vector.train <- append(threshold.vector.train, threshold09)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix09[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix09[["byClass"]][["Recall"]])}

# Model 10
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain10 <- predict(cv_model_10, newdata=AOSI.training, type="prob")
rocCurve10 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain10[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc10 <- auc(rocCurve10)
plot(rocCurve10, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest10 <- predict(cv_model_10, AOSI.training, type = "prob")
threshold10 <- 0.246
pred10      <- factor( ifelse(probsTest10[, "ASD"] > threshold10, "ASD", "N") )
pred10      <- relevel(pred10, "ASD")    
Optimal.tuningMatrix10 <- confusionMatrix(pred10, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc10)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 10")
  threshold.vector.train <- append(threshold.vector.train, threshold10)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix10[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix10[["byClass"]][["Recall"]])}

# Model 11
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain11 <- predict(cv_model_11, newdata=AOSI.training, type="prob")
rocCurve11 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain11[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc11 <- auc(rocCurve11)
plot(rocCurve11, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest11 <- predict(cv_model_11, AOSI.training, type = "prob")
threshold11 <- 0.387
pred11      <- factor( ifelse(probsTest11[, "ASD"] > threshold11, "ASD", "N") )
pred11      <- relevel(pred11, "ASD")    
Optimal.tuningMatrix11 <- confusionMatrix(pred11, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc11)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 11")
  threshold.vector.train <- append(threshold.vector.train, threshold11)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix11[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix11[["byClass"]][["Recall"]])}

# Model 12
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain12 <- predict(cv_model_12, newdata=AOSI.training, type="prob")
rocCurve12 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain12[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc12 <- auc(rocCurve12)
plot(rocCurve12, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest12 <- predict(cv_model_12, AOSI.training, type = "prob")
threshold12 <- 0.243
pred12      <- factor( ifelse(probsTest12[, "ASD"] > threshold12, "ASD", "N") )
pred12      <- relevel(pred12, "ASD")    
Optimal.tuningMatrix12 <- confusionMatrix(pred12, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc12)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 12")
  threshold.vector.train <- append(threshold.vector.train, threshold12)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix12[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix12[["byClass"]][["Recall"]])}

# Model 13
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain13 <- predict(cv_model_13, newdata=AOSI.training, type="prob")
rocCurve13 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain13[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc13 <- auc(rocCurve13)
plot(rocCurve13, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest13 <- predict(cv_model_13, AOSI.training, type = "prob")
threshold13 <- 0.373
pred13      <- factor( ifelse(probsTest13[, "ASD"] > threshold13, "ASD", "N") )
pred13      <- relevel(pred13, "ASD")    
Optimal.tuningMatrix13 <- confusionMatrix(pred13, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc13)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 13")
  threshold.vector.train <- append(threshold.vector.train, threshold13)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix13[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix13[["byClass"]][["Recall"]])}

# Model 14
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain14 <- predict(cv_model_14, newdata=AOSI.training, type="prob")
rocCurve14 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain14[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc14 <- auc(rocCurve14)
plot(rocCurve14, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest14 <- predict(cv_model_14, AOSI.training, type = "prob")
threshold14 <- 0.258
pred14      <- factor( ifelse(probsTest14[, "ASD"] > threshold14, "ASD", "N") )
pred14      <- relevel(pred14, "ASD")    
Optimal.tuningMatrix14 <- confusionMatrix(pred14, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc14)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 14")
  threshold.vector.train <- append(threshold.vector.train, threshold14)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix14[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix14[["byClass"]][["Recall"]])}

# Model 15
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain15 <- predict(cv_model_15, newdata=AOSI.training, type="prob")
rocCurve15 <- roc(response = AOSI.training$dx36,
                  predictor = probsTrain15[,"ASD"],
                  levels = rev(levels(AOSI.training$dx36)))
auc15 <- auc(rocCurve15)
plot(rocCurve15, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest15 <- predict(cv_model_15, AOSI.training, type = "prob")
threshold15 <- 0.361
pred15      <- factor( ifelse(probsTest15[, "ASD"] > threshold15, "ASD", "N") )
pred15      <- relevel(pred15, "ASD")    
Optimal.tuningMatrix15 <- confusionMatrix(pred15, AOSI.training$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.train <- append(auc.vector.train, auc15)
  number.vector.train <- append(number.vector.train, "Optimal RegLogistic - TRAIN- Model 15")
  threshold.vector.train <- append(threshold.vector.train, threshold15)
  accuracy.vector.train  <- append(accuracy.vector.train, Optimal.tuningMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.train <- append(accuracy.99CI.lower.vector.train, Optimal.tuningMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.train <- append(accuracy.99CI.upper.vector.train, Optimal.tuningMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.train <- append(Kappa.vector.train, Optimal.tuningMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.train <- append(McnemarP.vector.train, Optimal.tuningMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.train <- append(sensitivity.vector.train, Optimal.tuningMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.train <- append(specificity.vector.train, Optimal.tuningMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.train <- append(PPV.vector.train, Optimal.tuningMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.train <- append(NPV.vector.train, Optimal.tuningMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.train <- append(precision.vector.train, Optimal.tuningMatrix15[["byClass"]][["Precision"]])
  recall.vector.train <- append(recall.vector.train,Optimal.tuningMatrix15[["byClass"]][["Recall"]])}

# Extract data from vectors into a single data frame for ease of assessing results
Extracted.optimal.train.data <- data.frame(number.vector.train, threshold.vector.train, accuracy.vector.train, accuracy.99CI.lower.vector.train, accuracy.99CI.upper.vector.train, Kappa.vector.train, McnemarP.vector.train, auc.vector.train, sensitivity.vector.train, specificity.vector.train, PPV.vector.train, NPV.vector.train, precision.vector.train, recall.vector.train)
Extracted.optimal.train.data

# Exports the data frame as a comma delimited file 
write.table(Extracted.optimal.train.data, file="Optimized Combined Results - RegLogistic TL10 - WITHOUT gender on TRAIN data.csv", sep=",")


# Logistic models applied to and optimized for **TESTING** data ----

# Generation of vectors to store variables in prior to generating data frames
{ auc.vector.test <- c("AUC")
  number.vector.test <- c("Model#")
  threshold.vector.test <- c("Threshold")
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

# MODEL 1
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain01 <- predict(cv_model_01, newdata=AOSI.testing, type="prob")
rocCurve01 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain01[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc01 <- auc(rocCurve01)
plot(rocCurve01, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest01 <- predict(cv_model_01, AOSI.testing, type = "prob")
threshold01 <- 0.166
pred01      <- factor( ifelse(probsTest01[, "ASD"] > threshold01, "ASD", "N") )
pred01      <- relevel(pred01, "ASD")    
Optimal.tuningMatrix01 <- confusionMatrix(pred01, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc01)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model 01")
  threshold.vector.test <- append(threshold.vector.test, threshold01)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix01[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix01[["byClass"]][["Recall"]])}

# MODEL 2
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain02 <- predict(cv_model_02, newdata=AOSI.testing, type="prob")
rocCurve02 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain02[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc02 <- auc(rocCurve02)
plot(rocCurve02, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest02 <- predict(cv_model_02, AOSI.testing, type = "prob")
threshold02 <- 0.166
pred02      <- factor( ifelse(probsTest02[, "ASD"] > threshold02, "ASD", "N") )
pred02      <- relevel(pred02, "ASD")    
Optimal.tuningMatrix02 <- confusionMatrix(pred02, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc02)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  02")
  threshold.vector.test <- append(threshold.vector.test, threshold02)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix02[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix02[["byClass"]][["Recall"]])}

# MODEL 3
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain03 <- predict(cv_model_03, newdata=AOSI.testing, type="prob")
rocCurve03 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain03[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc03 <- auc(rocCurve03)
plot(rocCurve03, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest03 <- predict(cv_model_03, AOSI.testing, type = "prob")
threshold03 <- 0.194
pred03      <- factor( ifelse(probsTest03[, "ASD"] > threshold03, "ASD", "N") )
pred03      <- relevel(pred03, "ASD")    
Optimal.tuningMatrix03 <- confusionMatrix(pred03, AOSI.testing$dx36)


# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc03)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  03")
  threshold.vector.test <- append(threshold.vector.test, threshold03)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix03[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix03[["byClass"]][["Recall"]])}

# Model 4
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain04 <- predict(cv_model_04, newdata=AOSI.testing, type="prob")
rocCurve04 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain04[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc04 <- auc(rocCurve04)
plot(rocCurve04, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest04 <- predict(cv_model_04, AOSI.testing, type = "prob")
threshold04 <- 0.177
pred04      <- factor( ifelse(probsTest04[, "ASD"] > threshold04, "ASD", "N") )
pred04      <- relevel(pred04, "ASD")    
Optimal.tuningMatrix04 <- confusionMatrix(pred04, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc04)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  04")
  threshold.vector.test <- append(threshold.vector.test, threshold04)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix04[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix04[["byClass"]][["Recall"]])}

# Model 5
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain05 <- predict(cv_model_05, newdata=AOSI.testing, type="prob")
rocCurve05 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain05[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc05 <- auc(rocCurve05)
plot(rocCurve05, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest05 <- predict(cv_model_05, AOSI.testing, type = "prob")
threshold05 <- 0.226
pred05      <- factor( ifelse(probsTest05[, "ASD"] > threshold05, "ASD", "N") )
pred05      <- relevel(pred05, "ASD")    
Optimal.tuningMatrix05 <- confusionMatrix(pred05, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc05)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  05")
  threshold.vector.test <- append(threshold.vector.test, threshold05)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix05[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix05[["byClass"]][["Recall"]])}

# Model 6
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain06 <- predict(cv_model_06, newdata=AOSI.testing, type="prob")
rocCurve06 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain06[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc06 <- auc(rocCurve06)
plot(rocCurve06, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest06 <- predict(cv_model_06, AOSI.testing, type = "prob")
threshold06 <- 0.325
pred06      <- factor( ifelse(probsTest06[, "ASD"] > threshold06, "ASD", "N") )
pred06      <- relevel(pred06, "ASD")    
Optimal.tuningMatrix06 <- confusionMatrix(pred06, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc06)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  06")
  threshold.vector.test <- append(threshold.vector.test, threshold06)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix06[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix06[["byClass"]][["Recall"]])}

# Model 7
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain07 <- predict(cv_model_07, newdata=AOSI.testing, type="prob")
rocCurve07 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain07[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc07 <- auc(rocCurve07)
plot(rocCurve07, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest07 <- predict(cv_model_07, AOSI.testing, type = "prob")
threshold07 <- 0.175
pred07      <- factor( ifelse(probsTest07[, "ASD"] > threshold07, "ASD", "N") )
pred07      <- relevel(pred07, "ASD")    
Optimal.tuningMatrix07 <- confusionMatrix(pred07, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc07)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  07")
  threshold.vector.test <- append(threshold.vector.test, threshold07)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix07[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix07[["byClass"]][["Recall"]])}

# Model 8
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain08 <- predict(cv_model_08, newdata=AOSI.testing, type="prob")
rocCurve08 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain08[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc08 <- auc(rocCurve08)
plot(rocCurve08, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest08 <- predict(cv_model_08, AOSI.testing, type = "prob")
threshold08 <- 0.300
pred08      <- factor( ifelse(probsTest08[, "ASD"] > threshold08, "ASD", "N") )
pred08      <- relevel(pred08, "ASD")    
Optimal.tuningMatrix08 <- confusionMatrix(pred08, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc08)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  08")
  threshold.vector.test <- append(threshold.vector.test, threshold08)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix08[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix08[["byClass"]][["Recall"]])}

# Model 9
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain09 <- predict(cv_model_09, newdata=AOSI.testing, type="prob")
rocCurve09 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain09[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc09 <- auc(rocCurve09)
plot(rocCurve09, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest09 <- predict(cv_model_09, AOSI.testing, type = "prob")
threshold09 <- 0.183
pred09      <- factor( ifelse(probsTest09[, "ASD"] > threshold09, "ASD", "N") )
pred09      <- relevel(pred09, "ASD")    
Optimal.tuningMatrix09 <- confusionMatrix(pred09, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc09)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  09")
  threshold.vector.test <- append(threshold.vector.test, threshold09)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix09[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix09[["byClass"]][["Recall"]])}

# Model 10
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain10 <- predict(cv_model_10, newdata=AOSI.testing, type="prob")
rocCurve10 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain10[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc10 <- auc(rocCurve10)
plot(rocCurve10, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest10 <- predict(cv_model_10, AOSI.testing, type = "prob")
threshold10 <- 0.227
pred10      <- factor( ifelse(probsTest10[, "ASD"] > threshold10, "ASD", "N") )
pred10      <- relevel(pred10, "ASD")    
Optimal.tuningMatrix10 <- confusionMatrix(pred10, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc10)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  10")
  threshold.vector.test <- append(threshold.vector.test, threshold10)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix10[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix10[["byClass"]][["Recall"]])}

# Model 11
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain11 <- predict(cv_model_11, newdata=AOSI.testing, type="prob")
rocCurve11 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain11[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc11 <- auc(rocCurve11)
plot(rocCurve11, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest11 <- predict(cv_model_11, AOSI.testing, type = "prob")
threshold11 <- 0.226
pred11      <- factor( ifelse(probsTest11[, "ASD"] > threshold11, "ASD", "N") )
pred11      <- relevel(pred11, "ASD")    
Optimal.tuningMatrix11 <- confusionMatrix(pred11, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc11)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  11")
  threshold.vector.test <- append(threshold.vector.test, threshold11)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix11[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix11[["byClass"]][["Recall"]])}

# Model 12
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain12 <- predict(cv_model_12, newdata=AOSI.testing, type="prob")
rocCurve12 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain12[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc12 <- auc(rocCurve12)
plot(rocCurve12, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest12 <- predict(cv_model_12, AOSI.testing, type = "prob")
threshold12 <- 0.330
pred12      <- factor( ifelse(probsTest12[, "ASD"] > threshold12, "ASD", "N") )
pred12      <- relevel(pred12, "ASD")    
Optimal.tuningMatrix12 <- confusionMatrix(pred12, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc12)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  12")
  threshold.vector.test <- append(threshold.vector.test, threshold12)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix12[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix12[["byClass"]][["Recall"]])}

# Model 13
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain13 <- predict(cv_model_13, newdata=AOSI.testing, type="prob")
rocCurve13 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain13[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc13 <- auc(rocCurve13)
plot(rocCurve13, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest13 <- predict(cv_model_13, AOSI.testing, type = "prob")
threshold13 <- 0.184
pred13      <- factor( ifelse(probsTest13[, "ASD"] > threshold13, "ASD", "N") )
pred13      <- relevel(pred13, "ASD")    
Optimal.tuningMatrix13 <- confusionMatrix(pred13, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc13)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  13")
  threshold.vector.test <- append(threshold.vector.test, threshold13)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix13[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix13[["byClass"]][["Recall"]])}

# Model 14
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain14 <- predict(cv_model_14, newdata=AOSI.testing, type="prob")
rocCurve14 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain14[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc14 <- auc(rocCurve14)
plot(rocCurve14, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest14 <- predict(cv_model_14, AOSI.testing, type = "prob")
threshold14 <- 0.235
pred14      <- factor( ifelse(probsTest14[, "ASD"] > threshold14, "ASD", "N") )
pred14      <- relevel(pred14, "ASD")    
Optimal.tuningMatrix14 <- confusionMatrix(pred14, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc14)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  14")
  threshold.vector.test <- append(threshold.vector.test, threshold14)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix14[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix14[["byClass"]][["Recall"]])}

# Model 15
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain15 <- predict(cv_model_15, newdata=AOSI.testing, type="prob")
rocCurve15 <- roc(response = AOSI.testing$dx36,
                  predictor = probsTrain15[,"ASD"],
                  levels = rev(levels(AOSI.testing$dx36)))
auc15 <- auc(rocCurve15)
plot(rocCurve15, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest15 <- predict(cv_model_15, AOSI.testing, type = "prob")
threshold15 <- 0.226
pred15      <- factor( ifelse(probsTest15[, "ASD"] > threshold15, "ASD", "N") )
pred15      <- relevel(pred15, "ASD")    
Optimal.tuningMatrix15 <- confusionMatrix(pred15, AOSI.testing$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.test <- append(auc.vector.test, auc15)
  number.vector.test <- append(number.vector.test, "Optimal RegLogistic - TEST - Model  15")
  threshold.vector.test <- append(threshold.vector.test, threshold15)
  accuracy.vector.test  <- append(accuracy.vector.test, Optimal.tuningMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.test <- append(accuracy.99CI.lower.vector.test, Optimal.tuningMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.test <- append(accuracy.99CI.upper.vector.test, Optimal.tuningMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.test <- append(Kappa.vector.test, Optimal.tuningMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.test <- append(McnemarP.vector.test, Optimal.tuningMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.test <- append(sensitivity.vector.test, Optimal.tuningMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.test <- append(specificity.vector.test, Optimal.tuningMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.test <- append(PPV.vector.test, Optimal.tuningMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.test <- append(NPV.vector.test, Optimal.tuningMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.test <- append(precision.vector.test, Optimal.tuningMatrix15[["byClass"]][["Precision"]])
  recall.vector.test <- append(recall.vector.test,Optimal.tuningMatrix15[["byClass"]][["Recall"]])}

# # Extract data from vectors into a single data frame for ease of assessing results
Extracted.optimal.test.data <- data.frame(number.vector.test, threshold.vector.test, accuracy.vector.test, accuracy.99CI.lower.vector.test, accuracy.99CI.upper.vector.test, Kappa.vector.test, McnemarP.vector.test, auc.vector.test, sensitivity.vector.test, specificity.vector.test, PPV.vector.test, NPV.vector.test, precision.vector.test, recall.vector.test)
Extracted.optimal.test.data

# Exports the data frame as a comma delimited file 
write.table(Extracted.optimal.test.data, file="Optimized Combined Results - RegLogistic TL10 - WITHOUT gender on TEST data.csv", sep=",")


# Logistic models applied to and optimized for **INDEPENDENT** data ----

# Generation of vectors to store variables in prior to generating data frames
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
  recall.vector.independent <- c("Recall")}

# MODEL 1
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain01 <- predict(cv_model_01, newdata=AOSI.independent, type="prob")
rocCurve01 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain01[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc01 <- auc(rocCurve01)
plot(rocCurve01, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest01 <- predict(cv_model_01, AOSI.independent, type = "prob")
threshold01 <- 0.181
pred01      <- factor( ifelse(probsTest01[, "ASD"] > threshold01, "ASD", "N") )
pred01      <- relevel(pred01, "ASD")    
Optimal.tuningMatrix01 <- confusionMatrix(pred01, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc01)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 01")
  threshold.vector.independent <- append(threshold.vector.independent, threshold01)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix01[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix01[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix01[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix01[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix01[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix01[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix01[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix01[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix01[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix01[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix01[["byClass"]][["Recall"]])
  
  rm(auc01, probsTrain01, rocCurve01, probsTest01, pred01, threshold01, Optimal.tuningMatrix01)
}

# MODEL 2
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain02 <- predict(cv_model_02, newdata=AOSI.independent, type="prob")
rocCurve02 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain02[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc02 <- auc(rocCurve02)
plot(rocCurve02, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest02 <- predict(cv_model_02, AOSI.independent, type = "prob")
threshold02 <- 0.181
pred02      <- factor( ifelse(probsTest02[, "ASD"] > threshold02, "ASD", "N") )
pred02      <- relevel(pred02, "ASD")    
Optimal.tuningMatrix02 <- confusionMatrix(pred02, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc02)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 02")
  threshold.vector.independent <- append(threshold.vector.independent, threshold02)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix02[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix02[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix02[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix02[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix02[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix02[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix02[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix02[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix02[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix02[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix02[["byClass"]][["Recall"]])
  
  rm(auc02, probsTrain02, rocCurve02, probsTest02, pred02, threshold02, Optimal.tuningMatrix02)
}

# MODEL 3
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain03 <- predict(cv_model_03, newdata=AOSI.independent, type="prob")
rocCurve03 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain03[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc03 <- auc(rocCurve03)
plot(rocCurve03, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest03 <- predict(cv_model_03, AOSI.independent, type = "prob")
threshold03 <- 0.255
pred03      <- factor( ifelse(probsTest03[, "ASD"] > threshold03, "ASD", "N") )
pred03      <- relevel(pred03, "ASD")    
Optimal.tuningMatrix03 <- confusionMatrix(pred03, AOSI.independent$dx36)


# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc03)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 03")
  threshold.vector.independent <- append(threshold.vector.independent, threshold03)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix03[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix03[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix03[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix03[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix03[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix03[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix03[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix03[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix03[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix03[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix03[["byClass"]][["Recall"]])
  
  rm(auc03, probsTrain03, rocCurve03, probsTest03, pred03, threshold03, Optimal.tuningMatrix03)
}

# Model 4
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain04 <- predict(cv_model_04, newdata=AOSI.independent, type="prob")
rocCurve04 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain04[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc04 <- auc(rocCurve04)
plot(rocCurve04, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest04 <- predict(cv_model_04, AOSI.independent, type = "prob")
threshold04 <- 0.255
pred04      <- factor( ifelse(probsTest04[, "ASD"] > threshold04, "ASD", "N") )
pred04      <- relevel(pred04, "ASD")    
Optimal.tuningMatrix04 <- confusionMatrix(pred04, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc04)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 04")
  threshold.vector.independent <- append(threshold.vector.independent, threshold04)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix04[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix04[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix04[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix04[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix04[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix04[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix04[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix04[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix04[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix04[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix04[["byClass"]][["Recall"]])
  
  rm(auc04, probsTrain04, rocCurve04, probsTest04, pred04, threshold04, Optimal.tuningMatrix04)
}

# Model 5
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain05 <- predict(cv_model_05, newdata=AOSI.independent, type="prob")
rocCurve05 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain05[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc05 <- auc(rocCurve05)
plot(rocCurve05, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest05 <- predict(cv_model_05, AOSI.independent, type = "prob")
threshold05 <- 0.212
pred05      <- factor( ifelse(probsTest05[, "ASD"] > threshold05, "ASD", "N") )
pred05      <- relevel(pred05, "ASD")    
Optimal.tuningMatrix05 <- confusionMatrix(pred05, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc05)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 05")
  threshold.vector.independent <- append(threshold.vector.independent, threshold05)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix05[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix05[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix05[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix05[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix05[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix05[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix05[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix05[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix05[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix05[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix05[["byClass"]][["Recall"]])
  
  rm(auc05, probsTrain05, rocCurve05, probsTest05, pred05, threshold05, Optimal.tuningMatrix05)
}

# Model 6
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain06 <- predict(cv_model_06, newdata=AOSI.independent, type="prob")
rocCurve06 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain06[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc06 <- auc(rocCurve06)
plot(rocCurve06, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest06 <- predict(cv_model_06, AOSI.independent, type = "prob")
threshold06 <- 0.371
pred06      <- factor( ifelse(probsTest06[, "ASD"] > threshold06, "ASD", "N") )
pred06      <- relevel(pred06, "ASD")    
Optimal.tuningMatrix06 <- confusionMatrix(pred06, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc06)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 06")
  threshold.vector.independent <- append(threshold.vector.independent, threshold06)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix06[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix06[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix06[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix06[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix06[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix06[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix06[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix06[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix06[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix06[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix06[["byClass"]][["Recall"]])
  
  rm(auc06, probsTrain06, rocCurve06, probsTest06, pred06, threshold06, Optimal.tuningMatrix06)
}

# Model 7
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain07 <- predict(cv_model_07, newdata=AOSI.independent, type="prob")
rocCurve07 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain07[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc07 <- auc(rocCurve07)
plot(rocCurve07, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest07 <- predict(cv_model_07, AOSI.independent, type = "prob")
threshold07 <- 0.224
pred07      <- factor( ifelse(probsTest07[, "ASD"] > threshold07, "ASD", "N") )
pred07      <- relevel(pred07, "ASD")    
Optimal.tuningMatrix07 <- confusionMatrix(pred07, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc07)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 07")
  threshold.vector.independent <- append(threshold.vector.independent, threshold07)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix07[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix07[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix07[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix07[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix07[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix07[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix07[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix07[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix07[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix07[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix07[["byClass"]][["Recall"]])
  
  rm(auc07, probsTrain07, rocCurve07, probsTest07, pred07, threshold07, Optimal.tuningMatrix07)
}

# Model 8
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain08 <- predict(cv_model_08, newdata=AOSI.independent, type="prob")
rocCurve08 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain08[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc08 <- auc(rocCurve08)
plot(rocCurve08, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest08 <- predict(cv_model_08, AOSI.independent, type = "prob")
threshold08 <- 0.210
pred08      <- factor( ifelse(probsTest08[, "ASD"] > threshold08, "ASD", "N") )
pred08      <- relevel(pred08, "ASD")    
Optimal.tuningMatrix08 <- confusionMatrix(pred08, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc08)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 08")
  threshold.vector.independent <- append(threshold.vector.independent, threshold08)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix08[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix08[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix08[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix08[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix08[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix08[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix08[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix08[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix08[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix08[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix08[["byClass"]][["Recall"]])
  
  rm(auc08, probsTrain08, rocCurve08, probsTest08, pred08, threshold08, Optimal.tuningMatrix08)
}

# Model 9
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain09 <- predict(cv_model_09, newdata=AOSI.independent, type="prob")
rocCurve09 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain09[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc09 <- auc(rocCurve09)
plot(rocCurve09, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest09 <- predict(cv_model_09, AOSI.independent, type = "prob")
threshold09 <- 0.242
pred09      <- factor( ifelse(probsTest09[, "ASD"] > threshold09, "ASD", "N") )
pred09      <- relevel(pred09, "ASD")    
Optimal.tuningMatrix09 <- confusionMatrix(pred09, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc09)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 09")
  threshold.vector.independent <- append(threshold.vector.independent, threshold09)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix09[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix09[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix09[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix09[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix09[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix09[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix09[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix09[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix09[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix09[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix09[["byClass"]][["Recall"]])
  
  rm(auc09, probsTrain09, rocCurve09, probsTest09, pred09, threshold09, Optimal.tuningMatrix09)
}

# Model 10
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain10 <- predict(cv_model_10, newdata=AOSI.independent, type="prob")
rocCurve10 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain10[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc10 <- auc(rocCurve10)
plot(rocCurve10, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest10 <- predict(cv_model_10, AOSI.independent, type = "prob")
threshold10 <- 0.182
pred10      <- factor( ifelse(probsTest10[, "ASD"] > threshold10, "ASD", "N") )
pred10      <- relevel(pred10, "ASD")    
Optimal.tuningMatrix10 <- confusionMatrix(pred10, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc10)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 10")
  threshold.vector.independent <- append(threshold.vector.independent, threshold10)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix10[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix10[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix10[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix10[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix10[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix10[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix10[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix10[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix10[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix10[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix10[["byClass"]][["Recall"]])
  
  rm(auc10, probsTrain10, rocCurve10, probsTest10, pred10, threshold10, Optimal.tuningMatrix10)
}

# Model 11
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain11 <- predict(cv_model_11, newdata=AOSI.independent, type="prob")
rocCurve11 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain11[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc11 <- auc(rocCurve11)
plot(rocCurve11, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest11 <- predict(cv_model_11, AOSI.independent, type = "prob")
threshold11 <- 0.231
pred11      <- factor( ifelse(probsTest11[, "ASD"] > threshold11, "ASD", "N") )
pred11      <- relevel(pred11, "ASD")    
Optimal.tuningMatrix11 <- confusionMatrix(pred11, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc11)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 11")
  threshold.vector.independent <- append(threshold.vector.independent, threshold11)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix11[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix11[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix11[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix11[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix11[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix11[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix11[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix11[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix11[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix11[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix11[["byClass"]][["Recall"]])
  
  rm(auc11, probsTrain11, rocCurve11, probsTest11, pred11, threshold11, Optimal.tuningMatrix11)
}

# Model 12
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain12 <- predict(cv_model_12, newdata=AOSI.independent, type="prob")
rocCurve12 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain12[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc12 <- auc(rocCurve12)
plot(rocCurve12, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest12 <- predict(cv_model_12, AOSI.independent, type = "prob")
threshold12 <- 0.216
pred12      <- factor( ifelse(probsTest12[, "ASD"] > threshold12, "ASD", "N") )
pred12      <- relevel(pred12, "ASD")    
Optimal.tuningMatrix12 <- confusionMatrix(pred12, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc12)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 12")
  threshold.vector.independent <- append(threshold.vector.independent, threshold12)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix12[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix12[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix12[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix12[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix12[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix12[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix12[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix12[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix12[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix12[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix12[["byClass"]][["Recall"]])
  
  rm(auc12, probsTrain12, rocCurve12, probsTest12, pred12, threshold12, Optimal.tuningMatrix12)
}

# Model 13
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain13 <- predict(cv_model_13, newdata=AOSI.independent, type="prob")
rocCurve13 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain13[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc13 <- auc(rocCurve13)
plot(rocCurve13, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest13 <- predict(cv_model_13, AOSI.independent, type = "prob")
threshold13 <- 0.232
pred13      <- factor( ifelse(probsTest13[, "ASD"] > threshold13, "ASD", "N") )
pred13      <- relevel(pred13, "ASD")    
Optimal.tuningMatrix13 <- confusionMatrix(pred13, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc13)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 13")
  threshold.vector.independent <- append(threshold.vector.independent, threshold13)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix13[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix13[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix13[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix13[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix13[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix13[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix13[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix13[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix13[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix13[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix13[["byClass"]][["Recall"]])
  
  rm(auc13, probsTrain13, rocCurve13, probsTest13, pred13, threshold13, Optimal.tuningMatrix13)
}

# Model 14
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain14 <- predict(cv_model_14, newdata=AOSI.independent, type="prob")
rocCurve14 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain14[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc14 <- auc(rocCurve14)
plot(rocCurve14, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest14 <- predict(cv_model_14, AOSI.independent, type = "prob")
threshold14 <- 0.270
pred14      <- factor( ifelse(probsTest14[, "ASD"] > threshold14, "ASD", "N") )
pred14      <- relevel(pred14, "ASD")    
Optimal.tuningMatrix14 <- confusionMatrix(pred14, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc14)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 14")
  threshold.vector.independent <- append(threshold.vector.independent, threshold14)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix14[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix14[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix14[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix14[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix14[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix14[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix14[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix14[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix14[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix14[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix14[["byClass"]][["Recall"]])
  
  rm(auc14, probsTrain14, rocCurve14, probsTest14, pred14, threshold14, Optimal.tuningMatrix14)
}

# Model 15
# Find the logistic regression probability decision threshold with the highest combined sens/spec. 
probsTrain15 <- predict(cv_model_15, newdata=AOSI.independent, type="prob")
rocCurve15 <- roc(response = AOSI.independent$dx36,
                  predictor = probsTrain15[,"ASD"],
                  levels = rev(levels(AOSI.independent$dx36)))
auc15 <- auc(rocCurve15)
plot(rocCurve15, print.thres = "best")

# Manually set the optimum decision threshold and assess model performance
probsTest15 <- predict(cv_model_15, AOSI.independent, type = "prob")
threshold15 <- 0.180
pred15      <- factor( ifelse(probsTest15[, "ASD"] > threshold15, "ASD", "N") )
pred15      <- relevel(pred15, "ASD")    
Optimal.tuningMatrix15 <- confusionMatrix(pred15, AOSI.independent$dx36)

# Extract model performance statistics when the model uses the optimum decision boundary
{ auc.vector.independent <- append(auc.vector.independent, auc15)
  number.vector.independent <- append(number.vector.independent, "Optimal RegLogistic - INDEPENDENT - Model 15")
  threshold.vector.independent <- append(threshold.vector.independent, threshold15)
  accuracy.vector.independent  <- append(accuracy.vector.independent, Optimal.tuningMatrix15[["overall"]][["Accuracy"]])
  accuracy.99CI.lower.vector.independent <- append(accuracy.99CI.lower.vector.independent, Optimal.tuningMatrix15[["overall"]][["AccuracyLower"]])
  accuracy.99CI.upper.vector.independent <- append(accuracy.99CI.upper.vector.independent, Optimal.tuningMatrix15[["overall"]][["AccuracyUpper"]])
  Kappa.vector.independent <- append(Kappa.vector.independent, Optimal.tuningMatrix15[["overall"]][["Kappa"]])
  McnemarP.vector.independent <- append(McnemarP.vector.independent, Optimal.tuningMatrix15[["overall"]][["McnemarPValue"]])
  sensitivity.vector.independent <- append(sensitivity.vector.independent, Optimal.tuningMatrix15[["byClass"]][["Sensitivity"]])
  specificity.vector.independent <- append(specificity.vector.independent, Optimal.tuningMatrix15[["byClass"]][["Specificity"]])
  PPV.vector.independent <- append(PPV.vector.independent, Optimal.tuningMatrix15[["byClass"]][["Pos Pred Value"]])
  NPV.vector.independent <- append(NPV.vector.independent, Optimal.tuningMatrix15[["byClass"]][["Neg Pred Value"]])
  precision.vector.independent <- append(precision.vector.independent, Optimal.tuningMatrix15[["byClass"]][["Precision"]])
  recall.vector.independent <- append(recall.vector.independent,Optimal.tuningMatrix15[["byClass"]][["Recall"]])
  
  rm(auc15, probsTrain15, rocCurve15, probsTest15, pred15, threshold15, Optimal.tuningMatrix15)
}

# # Extract data from vectors into a single data frame for ease of assessing results
Extracted.optimal.independent.data <- data.frame(number.vector.independent, threshold.vector.independent, accuracy.vector.independent, accuracy.99CI.lower.vector.independent, accuracy.99CI.upper.vector.independent, Kappa.vector.independent, McnemarP.vector.independent, auc.vector.independent, sensitivity.vector.independent, specificity.vector.independent, PPV.vector.independent, NPV.vector.independent, precision.vector.independent, recall.vector.independent)
Extracted.optimal.independent.data

# Exports the data frame as a comma delimited file 
write.table(Extracted.optimal.independent.data, file="Optimized Combined Results - RegLogistic TL10- WITHOUT gender on INDEPENDENT data.csv", sep=",")
