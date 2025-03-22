library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(reshape2)
library(pROC)

dat <- read.csv("HR dataset.csv")

###### Variable conversion ######

numeric_vars <- c("Age", "DistanceFromHome", "EnvironmentSatisfaction", 
                  "JobSatisfaction", "MonthlyIncome", "TotalWorkingYears", 
                  "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", 
                  "YearsWithCurrManager")
dat[numeric_vars] <- lapply(dat[numeric_vars], as.numeric)

categorical_vars <- c("BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus")
dat[categorical_vars] <- lapply(dat[categorical_vars], as.factor)

binary_vars <- c("Attrition", "OverTime")
dat[binary_vars] <- lapply(dat[binary_vars], as.factor)


####### Exploratory Data Analysis ######

ggplot(dat, aes(x = Attrition, fill = Attrition)) +
  geom_bar() +
  labs(title = "Distribution of Attrition", x = "Attrition", y = "Count") +
  theme_minimal()

ggplot(dat, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Monthly Income by Attrition Status", x = "Attrition", y = "Monthly Income") +
  theme_minimal()

ggplot(dat, aes(x = BusinessTravel, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Rates Across Business Travel", x = "Business Travel", y = "Proportion") +
  theme_minimal()

# Correlation Heatmap
cor_matrix <- cor(dat[, numeric_vars], use = "pairwise.complete.obs")
cor_melt <- melt(cor_matrix)
ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap of Numeric Variables", x = "Variable", y = "Variable") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Pairwise Scatter plots
selected_vars <- dat[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome")]
pairs(selected_vars, col = ifelse(dat$Attrition == "Yes", "red", "blue"),
      main = "Pairplot of Selected Attributes by Attrition Status")

###### Data Partitioning and Balancing ######

yes_weight <- nrow(dat) / sum(dat$Attrition == "Yes")
dat$weights <- ifelse(dat$Attrition == "Yes", yes_weight, 1)
set.seed(1)
train_index <- createDataPartition(dat$Attrition, p = 0.7, list = FALSE)
training_data <- dat[train_index, ]
testing_data <- dat[-train_index, ]

###### Logistic Regression Model ######
logistic_model <- glm(Attrition ~ MonthlyIncome + YearsAtCompany + JobSatisfaction +
                        DistanceFromHome + OverTime * JobSatisfaction + BusinessTravel + Age,
                      data = training_data, family = binomial(link = "logit"),
                      weights = training_data$weights)
summary(logistic_model)

# Logistic Regression Prediction and Evaluation
logistic_model_probs <- predict(logistic_model, newdata = testing_data, type = "response")
logistic_model_preds <- ifelse(logistic_model_probs > 0.5, 1, 0)

evaluate_model <- function(actual, predicted, probs) {
  conf_matrix <- table(Predicted = predicted, Actual = actual)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  roc_curve <- roc(actual, probs)
  auc_value <- auc(roc_curve)
  list(Accuracy = accuracy, Precision = precision, Recall = recall, AUC = auc_value)
}
logistic_model_metrics <- evaluate_model(testing_data$Attrition, logistic_model_preds, logistic_model_probs)
logistic_model_metrics

# Adjusting Thresholds
thresholds <- seq(0.3, 0.7, by = 0.1)
threshold_metrics <- lapply(thresholds, function(th) {
  preds <- ifelse(logistic_model_probs > th, 1, 0)
  evaluate_model(testing_data$Attrition, preds, logistic_model_probs)
})
threshold_metrics_df <- do.call(rbind, threshold_metrics)
data.frame(Threshold = thresholds, threshold_metrics_df)

###### Decision Tree Model ######
decision_tree <- rpart(Attrition ~ MonthlyIncome + YearsAtCompany + JobSatisfaction +
                         DistanceFromHome + OverTime + BusinessTravel + Age,
                       data = training_data, method = "class", weights = training_data$weights)
rpart.plot(decision_tree, type = 4, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for Employee Attrition")

#Pruning
pruned_tree <- prune(decision_tree, cp = decision_tree$cptable[which.min(decision_tree$cptable[, "xerror"]), "CP"])
rpart.plot(pruned_tree, type = 4, extra = 101, fallen.leaves = TRUE, main = "Pruned Decision Tree")
importance <- as.data.frame(varImp(decision_tree))
importance

# Decision Tree Evaluation
dt_predictions <- predict(decision_tree, newdata = testing_data, type = "class")
binary_actual <- ifelse(testing_data$Attrition == "Yes", 1, 0)
binary_predictions <- ifelse(dt_predictions == "Yes", 1, 0)
dt_conf_matrix <- table(Predicted = binary_predictions, Actual = binary_actual)
dt_metrics <- list(
  Accuracy = sum(diag(dt_conf_matrix)) / sum(dt_conf_matrix),
  Precision = dt_conf_matrix[2, 2] / sum(dt_conf_matrix[2, ]),
  Recall = dt_conf_matrix[2, 2] / sum(dt_conf_matrix[, 2])
)
dt_metrics


###### K-Means Clustering ######
cluster_data <- training_data[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome", "Age")]
cluster_data_scaled <- scale(cluster_data)
set.seed(123)
kmeans_model <- kmeans(cluster_data_scaled, centers = 3, nstart = 25)
training_data$Cluster <- as.factor(kmeans_model$cluster)

pca <- prcomp(cluster_data_scaled, center = TRUE, scale. = TRUE)
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], Cluster = training_data$Cluster)
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "Clustering of Employees", x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

#Analysis
aggregate(training_data[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome", "Age")],
          by = list(Cluster = training_data$Cluster), FUN = mean)
