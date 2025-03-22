# Load necessary libraries
library(caret)       # For model training and data partitioning
library(ggplot2)     # For data visualization
library(rpart)       # For decision tree modeling
library(rpart.plot)  # For plotting decision trees
library(reshape2)    # For reshaping data (used in heatmap)
library(pROC)        # For ROC curve and AUC evaluation

# Load the dataset
dat <- read.csv("HR dataset.csv")

###### Step 1: Data Preparation ######
# Convert appropriate variables to numeric for analysis
numeric_vars <- c("Age", "DistanceFromHome", "EnvironmentSatisfaction", 
                  "JobSatisfaction", "MonthlyIncome", "TotalWorkingYears", 
                  "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", 
                  "YearsWithCurrManager")
dat[numeric_vars] <- lapply(dat[numeric_vars], as.numeric)

# Convert selected variables to categorical/factor type
categorical_vars <- c("BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus")
dat[categorical_vars] <- lapply(dat[categorical_vars], as.factor)

# Convert binary variables (Yes/No) to factors
binary_vars <- c("Attrition", "OverTime")
dat[binary_vars] <- lapply(dat[binary_vars], as.factor)


###### Step 2: Exploratory Data Analysis (EDA) ######
# 1. Distribution of Attrition
ggplot(dat, aes(x = Attrition, fill = Attrition)) +
  geom_bar() +
  labs(title = "Distribution of Attrition", x = "Attrition", y = "Count") +
  theme_minimal()

# 2. Monthly Income by Attrition Status
ggplot(dat, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Monthly Income by Attrition Status", x = "Attrition", y = "Monthly Income") +
  theme_minimal()

# 3. Attrition Rates Across Business Travel Categories
ggplot(dat, aes(x = BusinessTravel, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Rates Across Business Travel", x = "Business Travel", y = "Proportion") +
  theme_minimal()

# 4. Correlation Heatmap among Numeric Variables
cor_matrix <- cor(dat[, numeric_vars], use = "pairwise.complete.obs")
cor_melt <- melt(cor_matrix)
ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap of Numeric Variables") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5. Pairwise scatter plots for selected attributes, colored by Attrition
selected_vars <- dat[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome")]
pairs(selected_vars, col = ifelse(dat$Attrition == "Yes", "red", "blue"),
      main = "Pairplot of Selected Attributes by Attrition Status")


###### Step 3: Data Partitioning & Class Imbalance Handling ######
# Assign higher weight to the minority class ("Yes") to handle imbalance
yes_weight <- nrow(dat) / sum(dat$Attrition == "Yes")
dat$weights <- ifelse(dat$Attrition == "Yes", yes_weight, 1)

# Split data into training (70%) and testing (30%) sets
set.seed(100)
train_index <- createDataPartition(dat$Attrition, p = 0.7, list = FALSE)
training_data <- dat[train_index, ]
testing_data <- dat[-train_index, ]


###### Step 4: Logistic Regression Model ######
# Build logistic regression model with selected predictors and interaction term
logistic_model <- glm(Attrition ~ MonthlyIncome + YearsAtCompany + JobSatisfaction +
                        DistanceFromHome + OverTime * JobSatisfaction + BusinessTravel + Age,
                      data = training_data, family = binomial(link = "logit"),
                      weights = training_data$weights)

# Summarize model coefficients and significance
summary(logistic_model)

# Generate predicted probabilities for test set
logistic_model_probs <- predict(logistic_model, newdata = testing_data, type = "response")

# Classify based on default threshold (0.5)
logistic_model_preds <- ifelse(logistic_model_probs > 0.5, 1, 0)

# Function to evaluate classification performance
evaluate_model <- function(actual, predicted, probs) {
  conf_matrix <- table(Predicted = predicted, Actual = actual)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  roc_curve <- roc(actual, probs)
  auc_value <- auc(roc_curve)
  list(Accuracy = accuracy, Precision = precision, Recall = recall, AUC = auc_value)
}

# Evaluate performance on test set
logistic_model_metrics <- evaluate_model(testing_data$Attrition, logistic_model_preds, logistic_model_probs)
logistic_model_metrics

# Explore model performance at different classification thresholds
thresholds <- seq(0.3, 0.7, by = 0.1)
threshold_metrics <- lapply(thresholds, function(th) {
  preds <- ifelse(logistic_model_probs > th, 1, 0)
  evaluate_model(testing_data$Attrition, preds, logistic_model_probs)
})
threshold_metrics_df <- do.call(rbind, threshold_metrics)
data.frame(Threshold = thresholds, threshold_metrics_df)


###### Step 5: Decision Tree Model ######
# Build a decision tree classifier
decision_tree <- rpart(Attrition ~ MonthlyIncome + YearsAtCompany + JobSatisfaction +
                         DistanceFromHome + OverTime + BusinessTravel + Age,
                       data = training_data, method = "class", weights = training_data$weights)

# Visualize the full tree
rpart.plot(decision_tree, type = 4, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for Employee Attrition")

# Prune the tree to avoid overfitting (based on minimum cross-validation error)
pruned_tree <- prune(decision_tree, cp = decision_tree$cptable[which.min(decision_tree$cptable[, "xerror"]), "CP"])

# Visualize the pruned tree
rpart.plot(pruned_tree, type = 4, extra = 101, fallen.leaves = TRUE, main = "Pruned Decision Tree")

# Variable importance from the decision tree
importance <- as.data.frame(varImp(decision_tree))
importance

# Predict on test data and evaluate
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


###### Step 6: K-Means Clustering ######
# Select and scale numeric variables for clustering
cluster_data <- training_data[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome", "Age")]
cluster_data_scaled <- scale(cluster_data)

# Perform k-means clustering with 3 clusters
set.seed(100)
kmeans_model <- kmeans(cluster_data_scaled, centers = 3, nstart = 25)
training_data$Cluster <- as.factor(kmeans_model$cluster)

# Visualize clusters using PCA for dimensionality reduction
pca <- prcomp(cluster_data_scaled, center = TRUE, scale. = TRUE)
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], Cluster = training_data$Cluster)
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "Clustering of Employees", x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Analyze average attributes of each cluster group
aggregate(training_data[, c("MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome", "Age")],
          by = list(Cluster = training_data$Cluster), FUN = mean)
