---
title: "Black Friday"
output: html_notebook
---


```{r}
library(data.table)
library(ggplot2)
library(dplyr)
setwd("C:/Users/ankit/Desktop/R/black friday")
train <- fread("train_oSwQCTC/train.csv", stringsAsFactors = F)
test <- fread("test_HujdGe7/test.csv", stringsAsFactors = F)
test[['Purchase']] = NA
```

```{r}
bfsales <- rbindlist(list(train,test))
summary(bfsales)
```

```{r}
#Age vs Gender
ggplot(bfsales, aes(Age, fill = Gender)) + geom_bar()
```
```{r}
#Age vs City_Category
ggplot(bfsales, aes(Age, fill = City_Category)) + geom_bar()
```
```{r}
ggplot(bfsales, aes(Age, fill = factor(Marital_Status))) + geom_bar()
```

```{r}
bfsales$Gender <- ifelse(bfsales$Gender == 'M',1,0)
Age.list = c('0-17' = 0, '18-25' = 1, '26-35' = 2, '36-45' = 3, '46-50' = 4, '51-55' = 5, '55+' = 6)
bfsales$Age_Recoded <- Age.list[bfsales$Age]
bfsales$Stay_In_Current_City_Years[bfsales$Stay_In_Current_City_Years == '4+'] = '4'
bfsales[,Product_Category_2_NA := ifelse(sapply(bfsales$Product_Category_2, is.na) == TRUE,1,0)]
bfsales[,Product_Category_3_NA := ifelse(sapply(bfsales$Product_Category_3, is.na) == TRUE,1,0)]
bfsales[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999",  Product_Category_2)]
bfsales[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999",  Product_Category_3)]
```

```{r}
bfsales[, User_Count := .N, by = User_ID]
bfsales[, Product_Count := .N, by = Product_ID]
bfsales[, Mean_Purchase_Product := mean(Purchase, na.rm = T), by = Product_ID]
bfsales[, Mean_Purchase_User := mean(Purchase, na.rm = T), by = User_ID]
bfsales[, Mean_PP := mean(Purchase, na.rm = T), by = Product_Category_1]
bfsales$Mean_Purchase_Product <- ifelse(is.na(bfsales$Mean_Purchase_Product),bfsales$Mean_PP, bfsales$Mean_Purchase_Product)
```

```{r}
library(dummies)
bfsales <- dummy.data.frame(bfsales, names = c("City_Category"), sep = "_")
```

```{r}
sapply(bfsales, class)
library(purrr)
drop.cols <- c('Age')
bfsales.new <- bfsales[,!names(bfsales) %in% drop.cols]
```

```{r}

bfsales.new[,3:20] <- as.data.table(map(bfsales.new[,3:20], as.numeric))
bfsales.new$User_ID <- as.factor(bfsales$User_ID)
bfsales.new$Product_ID <- as.factor(bfsales$Product_ID)
```

```{r}
glimpse(bfsales.new)
```

```{r}
c.train <- bfsales.new[1:550068,]
c.test <- bfsales.new[-(1:550068),]
c.train <- c.train[c.train$Product_Category_1 <= 18,]

```

```{r}
library(h2o)
localH2O <- h2o.init(nthreads = -1)
 h2o.init()
```
```{r}
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)
```

```{r}
train.split <- h2o.splitFrame(train.h2o, ratios = 0.8, c('Sample_1','Sample_2'), seed = 1122)
```

```{r}
#GBM
system.time(
gbm.model <- h2o.gbm(y='Purchase', training_frame = train.h2o, ntrees = 1000, max_depth = 10, learn_rate = 0.01, seed = 1122))
```

```{r}
h2o.performance (gbm.model)
```

```{r}
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)
```

```{r}

system.time(
             dlearning.model <- h2o.deeplearning(y = 'Purchase',
             training_frame = train.h2o,
             epoch = 60,
             hidden = c(100,100),
             activation = "Rectifier",
             seed = 1122
             )
)
```

```{r}
h2o.performance(dlearning.model)
```

```{r}
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = (predict.dl2$predict))
write.csv(sub_dlearning, file = "sub_dlearning_new.csv", row.names = F)
```

```{r}
sub_ens <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = 0.4 * predict.dl2$predict + 0.6 * predict.gbm$predict)
write.csv(sub_ens, file = "sub_ensemble_new.csv", row.names = F)
```

```{r}
bfsales.new <- bfsales.new[,!(names(bfsales.new)) %in% c("User_ID", "Product_ID", "Purchase")]
```

```{r}
new_train <- bfsales.new[1:nrow(train),]
new_test <- bfsales.new[-(1:nrow(train)),]
y_train <- train$Purchase
library(xgboost)
dtrain <- xgb.DMatrix(as.matrix(new_train),label = y_train)
dtest <- xgb.DMatrix(as.matrix(new_test))

xgb_params = list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=0.8,
  eta=0.05,
  max_depth=10,
  subsample=0.8,
  seed=5,
  silent=TRUE)

bst <- xgb.train(data = dtrain, params = xgb_params,nround=698)

xgb.cv(xgb_params, dtrain, nrounds = 5000, nfold = 4, early_stopping_rounds = 100, print_every_n = 50)
```

```{r}
pred <- predict(bst,dtest)
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = pred)
write.csv(sub_dlearning, file = "sub_xgb_new.csv", row.names = F)
```

