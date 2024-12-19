library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(ROCR)

train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)
test <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = train) %>%
  update_role(ACTION, new_role = "outcome") %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.8)

svmlin <- svm_linear(cost = 10) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

linwf <- workflow() %>%
  add_model(svmlin) %>%
  add_recipe(my_recipe) %>%
  fit(data = train)

amazon_pred <- predict(linwf, new_data = test)

kaggle_submission <- test %>%
  select("id") %>%
  bind_cols(amazon_pred %>% select(".pred_class")) %>%
  rename(ACTION = .pred_class)

vroom_write(x = kaggle_submission, file = "./amazonPredictions.csv", delim = ",")

