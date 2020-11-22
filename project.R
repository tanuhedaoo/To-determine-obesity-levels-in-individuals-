# -------------------------------------------------------------------------------
# Data Cleaning
# -------------------------------------------------------------------------------
getwd()
obesity_data = read.csv("Obesity Dataset.csv")
head(obesity_data)
# first check if there are any missing values 
is.null(obesity_data)

# Replacing ouliers with NA
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}
obesity_data$Age = remove_outliers(obesity_data$Age)
# Removing rows containing NA values 
obesity_data = na.omit(obesity_data)
obesity_data = subset(obesity_data,select = -c(SMOKE,SCC))
obesity_data = obesity_data[!grepl("Always",obesity_data$CALC),]
obesity_data = obesity_data[!grepl("Bike",obesity_data$MTRANS),]
obesity_data = obesity_data[!grepl("Motorbike",obesity_data$MTRANS),]
obesity_data = obesity_data[!grepl("Always",obesity_data$CAEC),]
summary(obesity_data)

# -------------------------------------------------------------------------------
# FACTORS for multinomial
# ------------------------------------------------------------------------
obesity_data$family_history_with_overweight = factor(obesity_data$family_history_with_overweight,
                                           levels = c("no","yes"),
                                           labels = c(0,1))
obesity_data$FAVC = factor(obesity_data$FAVC,
                       levels = c("no","yes"),
                       labels = c(0,1))
obesity_data$Gender = factor(obesity_data$Gender,
                             levels = c("Male","Female"),
                             labels = c(1,0))
obesity_data$CAEC = factor(obesity_data$CAEC,
                           levels = c("no","Sometimes","Frequently"),
                           labels = c(1,2,3))
obesity_data$CALC = factor(obesity_data$CALC,
                           levels = c("no","Sometimes","Frequently"),
                           labels = c(1,2,3))
obesity_data$MTRANS = factor(obesity_data$MTRANS,
                             levels = c("Public_Transportation","Walking","Automobile"),
                             labels = c(1,2,3))
obesity_data$NObeyesdad = factor(obesity_data$NObeyesdad,
                                 levels = c("Insufficient_Weight",
                                            "Normal_Weight",
                                            "Overweight_Level_I",
                                            "Overweight_Level_II",
                                            "Obesity_Type_I",
                                            "Obesity_Type_II",
                                            "Obesity_Type_III"),
                                 labels = c(1,2,3,4,5,6,7))

# --------------------------------------------------------------------------------
# MODEL FITTING
# --------------------------------------------------------------------------------
library(nnet)
model = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight+
                  CALC + FAVC + FCVC + NCP + CAEC +  CH2O + FAF + TUE + MTRANS  ,
                 data = obesity_data)
step(model)
AIC(model)
BIC(model)
s = summary(model)
s
z =  s$coefficients/s$standard.errors
z
p_val = (1-pnorm(abs(z),0,1))*2
p_val
head(fitted(model))


# -------------------------------------------------------------------------------
# Ordinal logistic Regression
# -------------------------------------------------------------------------------
obesity_data$family_history_with_overweight=as.factor(obesity_data$family_history_with_overweight)
obesity_data$FAVC=as.factor(obesity_data$FAVC)
obesity_data$Gender=as.factor(obesity_data$Gender)
obesity_data$CAEC = factor(obesity_data$CAEC,
                           levels = c("no","Sometimes","Frequently"),
                           labels = c(1,2,3),
                           ordered = TRUE)
obesity_data$CALC = factor(obesity_data$CALC,
                           levels = c("no","Sometimes","Frequently"),
                           labels = c(1,2,3),
                           ordered = TRUE )
obesity_data$MTRANS = factor(obesity_data$MTRANS,
                             levels = c("Public_Transportation","Walking","Automobile"),
                             labels = c(1,2,3),
                             ordered = TRUE)
obesity_data$NObeyesdad = factor(obesity_data$NObeyesdad)

library(foreign)
library(ggplot2)
library(MASS)
library(Hmisc)
library(reshape2)
model = polr(NObeyesdad ~ Gender +  Age + Height + Weight + family_history_with_overweight +
               CALC + FAVC + FCVC + NCP + CAEC  ,
             Hess = TRUE,
             data = obesity_data,
             method = c("logistic"))
summary(model)
ctable = coef(summary(model))
p = pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable = cbind(ctable, "p value" = p)
ctable
ci = confint.default(model)
step(model)

# Since CH2O , FAF , TUE , MTRANS have p-value > 0.05, this means they are not significant.


library(caTools)
set.seed(123)
split = sample.split(obesity_data$NObeyesdad,SplitRatio = 0.8)
training_set = subset(obesity_data , split==TRUE)
test_set = subset(obesity_data , split == FALSE)

# feature scaling 

training_set[,2:4] = scale(training_set[,2:4])
test_set[,2:4] = scale(test_set[,2:4]) 
library(nnet)
library(caret)
library(tidyverse)

classifier = polr(NObeyesdad ~  CAEC  ,
             Hess = TRUE,
             data = obesity_data,
             method = c("logistic"))

classifier = polr(NObeyesdad ~ Age + Height + Weight + family_history_with_overweight+
                   FAVC + FCVC + NCP + CAEC + CALC , 
                 data = training_set,
                 Hess = TRUE,
                 method = c("logistic"))

# predictions

predictions = classifier %>% predict(test_set)
print(predictions)
cbind(predictions,test_set$NObeyesdad)

# accuracy 

mean(predictions == test_set$NObeyesdad)
ctable = coef(summary(classifier))
p = pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable = cbind(ctable, "p value" = p)
ctable
confint(classifier)
ci = confint.default(model)
corrplot(model)



# -------------------------------------------------------------------------------
# Support Vector Machines
# -------------------------------------------------------------------------------

library(e1071) 
obesity_data = read.csv("Obesity Dataset.csv")
n <- nrow(obesity_data)  # Number of observations
ntrain <- round(n*0.75)  # 75% for training set
set.seed(314)    # Set seed for reproducible results
tindex <- sample(n, ntrain)   # Create a random index
train_data <- obesity_data[tindex,]   # Create training set
test_data <- obesity_data[-tindex,]   # Create test set
svm1 <- svm(NObeyesdad ~ ., data=train_data, 
              method="C-classification", kernal="radial", 
              gamma=0.1, cost=10)
summary(svm1)
prediction <- predict(svm1, test_data)
xtab <- table(test_data$NObeyesdad, prediction)
xtab
acc = (58+64+83+81+89+55+78)/nrow(test_data)
acc

plot(svm1, train_data, Weight ~ Height,
     slice=list( Age = 25,family_history_with_overweight=0 , FAVC= 1,
                FCVC = 3,NCP = 1 ,SMOKE = 0,SCC = 1, CAEC = 0 ,  CH2O = 2 , FAF = 3,
                TUE = 2 , CALC = 1,
                MTRANS = 1))
confint(svm1)
step(svm1)
# classification error
err = 1.96*sqrt(((1-acc)*(acc))/nrow(test_data))
err
# True classification error of the model is between  2.17% and 5.41%

scores = c()
for(i in 1:100){
  indices = sample.int(n = 1000,replace = TRUE)
  sample = obesity_data[indices,]
  # calculate and store statistic
  statistic = mean(sample)
  scores[i] = statistic
}
scores

# -------------------------------------------------------------------------------
# LIKELIHOOD RATIO TEST 
# --------------------------------------------------------------------------------
library(nnet)
library(lmtest)
null_model = multinom(NObeyesdad ~ 1 ,
                 data = obesity_data,
                 family = binomial(link = logit),
                 trace = FALSE)
model_1 = multinom(NObeyesdad ~ Gender,
              data = obesity_data,
              family = binomial(link = logit),
              trace = FALSE)
lrtest(null_model,model_1)
# p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_1

model_1 = multinom(NObeyesdad ~ Gender ,
                      data = obesity_data,
                      family = binomial(link = logit),
                   trace = FALSE)
model_2 = multinom(NObeyesdad ~ Gender + Age,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = FALSE)
lrtest(model_1,model_2)

# p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_2


model_2 = multinom(NObeyesdad ~ Gender +Age ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_3 = multinom(NObeyesdad ~ Gender + Age + Height ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_2,model_3)
# # p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_3


model_3 = multinom(NObeyesdad ~ Gender +Age + Height,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_4 = multinom(NObeyesdad ~ Gender + Age + Height + Weight,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_3,model_4)
# # p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_4

model_4 = multinom(NObeyesdad ~ Gender + Age + Height + Weight ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_5 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_4,model_5)
# # p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_5

model_5 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_6 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FAVC,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_5,model_6)
# Here for alpha = 0.05 , we should reject the null hypothesis, but for alpha = 0.01, we should accept 
# Lets accept the null hypothesis and carry model_5 for next likelihood ratio test

model_6 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight 
                   + FAVC,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_7 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FCVC + FAVC,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_6,model_7)
# p-val is greater than 0.05 as well as 0.01, we should accept the null hypothesis and carry forward model_5

model_7 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FAVC + FCVC,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_8 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FAVC + FCVC + NCP,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_7,model_8)
#  p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_8

model_8 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FAVC + FCVC + NCP,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_9 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + FAVC + FCVC + NCP + CAEC ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_8,model_9)
#  p-value < 0.05 Reject null hypothesis , we should choose more complex model i.e model_9

model_8 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_10 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC + CH2O ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_8,model_10)
#  p-value > 0.05 Accept null hypothesis , we should move forward with model_9

model_8 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                    + NCP + FAVC + FCVC ,
                    data = obesity_data,
                    family = binomial(link = logit),
                   trace = F)
model_11 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                    + NCP + FAVC + FCVC + FAF ,
                    data = obesity_data,
                    family = binomial(link = logit),
                    trace = F)
lrtest(model_8,model_11)
#  p-value > 0.05 Accept null hypothesis , we should move forward with model_9

model_11 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF  ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_12 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF + TUE ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_11,model_12)
# Here for alpha = 0.05 , we should reject the null hypothesis, but for alpha = 0.01, we should accept 
# Lets accept the null hypothesis and carry model_5 for next likelihood ratio test

model_11 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF  ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_13 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF + CALC ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_11,model_13)
# Here for alpha = 0.05 , we should reject the null hypothesis, but for alpha = 0.01, we should accept 
# Lets accept the null hypothesis and carry model_5 for next likelihood ratio test

model_11 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF  ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
model_14 = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                   + NCP + FAVC + FCVC +FAF + MTRANS  ,
                   data = obesity_data,
                   family = binomial(link = logit),
                   trace = F)
lrtest(model_11,model_14)
# Here for alpha = 0.05 , we should reject the null hypothesis, but for alpha = 0.01, we should accept 
# Lets accept the null hypothesis and carry model_5 for next likelihood ratio test

final_model = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                       + NCP + FAVC + FCVC +FAF  ,
                       data = obesity_data,
                       family = binomial(link = logit),
                       trace = F)
library(caTools)
set.seed(123)
split = sample.split(obesity_data$NObeyesdad,SplitRatio = 0.8)
training_set = subset(obesity_data , split==TRUE)
test_set = subset(obesity_data , split == FALSE)

# feature scaling 

training_set[,2:4] = scale(training_set[,2:4])
test_set[,2:4] = scale(test_set[,2:4]) 

final_model = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                       + NCP + FAVC + FCVC +FAF  ,
                       data = training_set,
                       family = binomial(link = logit),
                       trace = F)
s = summary(final_model)
predictions = final_model %>% predict(test_set)
cbind(test_set$NObeyesdad,predictions)                         
mean(predictions == test_set$NObeyesdad)  
confusionMatrix(test_set$NObeyesdad,predictions)

# confidence intervals for all the parameters of final model
confint(final_model)

# correlation matrix
cor(obesity_data)

plot(full_model)



# -------------------------------------------------------------------------------
# multicollinearity 
# -------------------------------------------------------------------------------


# correlation plot - METHOD 1
model = multinom(NObeyesdad ~ .,data = obesity_data,trace = F)
dmy = dummyVars("~.",data = obesity_data)
data1 <- data.frame(predict(dmy, newdata = obesity_data))
corrplot::corrplot(cor(obesity_data))


# correlation plot - METHOD 2
obesity_data$family_history_with_overweight = as.numeric(obesity_data$family_history_with_overweight)
obesity_data$FAVC = as.numeric(obesity_data$FAVC)
obesity_data$Gender = as.numeric(obesity_data$Gender)
obesity_data$CAEC = as.numeric(obesity_data$CAEC)
obesity_data$CALC = as.numeric(obesity_data$CALC)
obesity_data$MTRANS = as.numeric(obesity_data$MTRANS)
obesity_data$NObeyesdad = as.numeric(obesity_data$NObeyesdad)
library(corpcor)
cor2pcor(cov(obesity_data))
cor(obesity_data)

install.packages("mctest")
library(mctest)
omcdiag(final_model)
imcdiag(final_model)
corrplot::corrplot(cor(obesity_data))
install.packages("GGally")
library(GGally)
ggpairs(obesity_data)
final_model = multinom(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                       + NCP +FAF  ,
                       data = obesity_data,
                       family = binomial(link = logit),
                       trace = F)
car::vif(final_model)

# -------------------------------------------------------------------------------
# LASSO and Ridge regression
# -------------------------------------------------------------------------------
library(glmnet)

obesity_data$NObeyesdad= as.numeric(obesity_data$NObeyesdad)
set.seed(123)
split = sample.split(obesity_data$NObeyesdad,SplitRatio = 0.8)
training_set = subset(obesity_data , split==TRUE)
test_set = subset(obesity_data , split == FALSE)
x_train = model.matrix(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                 + NCP + FAVC + FCVC +FAF , training_set)[,-1]
x_test = model.matrix(NObeyesdad ~ Gender + Age + Height + Weight + family_history_with_overweight
                       + NCP + FAVC + FCVC +FAF , test_set)[,-1]
y_train = (training_set$NObeyesdad-min(obesity_data$NObeyesdad))/(max(obesity_data$NObeyesdad)-min(obesity_data$NObeyesdad))
y_test = (test_set$NObeyesdad-min(obesity_data$NObeyesdad))/(max(obesity_data$NObeyesdad)-min(obesity_data$NObeyesdad))

cv.out = cv.glmnet(x_train,y_train,alpha = 1)
plot(cv.out)
best_lambda = cv.out$lambda.min
best_lambda
pred = predict(cv.out , s = best_lambda , x_test)
pred = as.vector(pred)
Metrics::mse(y_test,pred)
coef(cv.out,cv.out$lambda.min)
coef(cv.out,cv.out$lambda.1se)
pred = predict(cv.out , s = cv.out$lambda.1se , x_test)
pred = as.vector(pred)
Metrics::mse(y_test,pred)
summary(cv.out)
# -------------------------------------------------------------------------------
# AIC , BIC Selection
# -------------------------------------------------------------------------------

library(nnet)
model = multinom(NObeyesdad ~ Gender + Age + Height +Weight+ family_history_with_overweight+
                   FAVC + FCVC + NCP + CAEC +  CH2O + FAF +
                   TUE + MTRANS  ,data = obesity_data,trace = F)
summary(model)
s = step(model,trace = 0)
summary(s)
AIC(s)

install.packages("BMA")
library(BMA)
model_bic = bic.glm(NObeyesdad ~ Gender + Age + Height  + family_history_with_overweight+
  CALC + FAVC + FCVC + NCP + CAEC +  CH2O + FAF + TUE + MTRANS  ,glm.family = binomial(link = logit),
  data = obesity_data)
model_bic$postprob
model_bic$label
model_bic$probne0

#  "Age,family_history_with_overweight,CALC,FCVC,NCP,CAEC,TUE,MTRANS" 

model_aic = multinom(NObeyesdad ~ Gender + Age + Height + Weight +  family_history_with_overweight+
                  CALC + FAVC + FCVC + NCP + CAEC +  CH2O + FAF + TUE + MTRANS  ,family = binomial(link = logit),
                data = obesity_data)
summary(model_aic)
step_aic = step(model_aic)

# Gender + Age + Height + family_history_with_overweight + CALC + FCVC + NCP + CAEC + TUE + MTRANS
summary(step_aic)
confint(model)

library(leaps)
regfit.full = regsubsets(NObeyesdad ~ Gender + Age + Height  +  family_history_with_overweight+
                           CALC + FAVC + FCVC + NCP + CAEC +  CH2O + FAF + TUE + MTRANS , data = obesity_data ,nvmax = 17)
summary(regfit.full)
reg.summary = summary(regfit.full)
names(reg.summary)
reg.summary$rsq
plot(c(1:16),reg.summary$rsq , xlab = "Number of variable",ylab = "R- square")
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",type="l")
plot(reg.summary$adjr2 ,xlab="Number of Variables ", ylab="Adjusted RSq",type="l")
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type='l')


plot(regfit.full,scale="bic")
coef(regfit.full,8)

# -------------------------------------------------------------------------------
# Training Final Model
# -------------------------------------------------------------------------------
model_final = glm(NObeyesdad ~ .)
plot.roc(validate$real,validate$pred, col = "red", main="ROC Validation set",
         percent = TRUE, print.auc = TRUE)


