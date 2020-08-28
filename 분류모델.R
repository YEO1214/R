data("infert")
str(infert)
colSums(is.na(infert)) #결측치 없음
infert1<-infert
infert1$case<-as.factor(infert1$case)
Y<-infert1[,c("case")]
#str(Y)

#독립변수 추출, 표준화
infert2<-infert1[,c("age","parity","induced","spontaneous")]
infert_x<-scale(infert2)

data<-data.frame(Y,infert_x) #Y: case
str(data)

#종속변수 7:3으로 분할
set.seed(592)
library(caret)
part <- createDataPartition(data$Y, p=0.7)    
data.train <- data[part$Resample1, ] #훈련용 데이터
table(data.train$Y)
data.test<-data[-part$Resample1,] #테스티용 데이터
table(data.test$Y)

#나이브베이즈 모델
# 훈련용 데이터로 나이브 베이즈 모델을 생성하기
#install.packages("e1071")
library(e1071)
nai.fit <- naiveBayes(Y~., data=data.train)

# 테스트 데이터로 예측을 수행하고, 나이브 베이즈 모델의 성능 평가하기
#nai.pred1 <- predict(nai.fit, data.test)
#nai.tb1 <- table(nai.pred1, data.test$Y)
#nai.tb1
nai.pred <- predict(nai.fit, data.test[,-1], type='class')
nai.tb <- table(nai.pred, data.test$Y)
nai.tb

# 정분류율(accuracy)
mean(data.test$Y == nai.pred)      # accuracy
#[1] 0.7123288

# 오분류율(error rate) 
(1-sum(diag(nai.tb))/sum(nai.tb))      # error rate
#[1] 0.2876712

# ROC곡선, AUC 
library(ROCR)

# ROC : tpr & fpr 
nb.pred <- prediction(as.integer(nai.pred), as.integer(data.test$Y))
roc     <- performance(nb.pred, measure = "tpr", x.measure = "fpr")
roc.x = unlist(slot(roc, 'x.values'))
roc.y = unlist(slot(roc, 'y.values'))
plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
     ylab="True Positive Rate", xlab="False Positive Rate")
lines(x=roc.x, y=roc.y, col="orange", lwd=2)

# AUC (The Area Under an ROC Curve)
auc <- performance(nb.pred, measure="auc")
auc <- auc@y.values[[1]]
auc
#[1] 0.5943878

#로지스틱 회귀모델
glm.fit<-glm(Y~.,data=data.train, family=binomial(link="logit"))
summary(glm.fit)

# 테스트 데이터로 모델 성능 평가 수행하기
glm.probs <- predict(glm.fit, data.test, type="response")
glm.pred <- ifelse(glm.probs > .5, 1, 0)
table(data.test$Y, glm.pred)

mean(data.test$Y == glm.pred)      # accuracy
#[1] 0.7123288
mean(data.test$Y != glm.pred)      # error rate
#[1] 0.2876712


# ROC곡선와 AUC 확인하기 
library(ROCR)

# ROC : tpr & fpr 
glm.pred <- prediction(glm.probs, data.test$Y)
roc <- performance(glm.pred, measure = "tpr", x.measure = "fpr")
roc.x = unlist(slot(roc, 'x.values'))
roc.y = unlist(slot(roc, 'y.values'))
plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
     ylab="True Positive Rate", xlab="False Positive Rate")
lines(x=roc.x, y=roc.y, col="orange", lwd=2)

# AUC (The Area Under an ROC Curve)
auc <- performance(glm.pred, measure = "auc")
auc <- auc@y.values[[1]]
auc
#[1] 0.7971939
