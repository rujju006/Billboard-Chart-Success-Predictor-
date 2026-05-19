# Load libraries
suppressMessages(suppressWarnings({
  library(tidyverse)
  library(caret)
  library(ggplot2)
  library(qgraph)
  library(pROC)
  library(broom)
  library(gridExtra)
  library(tidyr)
  library(class)
  library(randomForest)
  library(gbm)
  library(MASS)
  library(dplyr)
  library(doParallel)
  library(ranger)
  library(fastshap)
}))

# 1. Data Preparation & Cleaning
spotify <- read.csv("Data/Hot 100 Audio Features.csv")
billboard <- read.csv("Data/Hot Stuff.csv")

# billboard and spotify data cleaning and merging
names(spotify)
names(billboard)
colSums(is.na(billboard)) %>% .[. > 0]   
colSums(is.na(spotify)) %>% .[. > 0]

billboard_peaks <- billboard %>%
  group_by(SongID, Performer) %>%
  summarise(
    peak_position = min(Peak.Position, na.rm = TRUE),
    .groups = "drop"
  )

spotify_clean <- spotify %>%
  drop_na(danceability:time_signature) %>%
  group_by(SongID, Performer) %>%
  summarise(
    danceability = mean(danceability),
    energy = mean(energy),
    valence = mean(valence),
    tempo = mean(tempo),
    loudness = mean(loudness),
    acousticness = mean(acousticness),
    instrumentalness = mean(instrumentalness),
    speechiness = mean(speechiness),
    liveness = mean(liveness),
    spotify_track_duration_ms = mean(spotify_track_duration_ms),
    key = first(key),
    mode = first(mode),
    time_signature = first(time_signature),
    .groups = "drop"
  )

colSums(is.na(spotify_clean)) %>% .[. > 0]  
song_level <- spotify_clean %>%
  inner_join(billboard_peaks, by = "SongID")

colSums(is.na(song_level)) %>% .[. > 0]  
song_level <- song_level %>% drop_na()

# created binary response variable
song_level <- song_level %>%
  mutate(hit = ifelse(peak_position <= 40, 1, 0)) %>%
  dplyr::select(-peak_position)

# Audio features
audio_features <- c(
  "danceability", "energy", "valence", "tempo", "loudness",
  "acousticness", "instrumentalness", "speechiness", "liveness",
  "spotify_track_duration_ms", "key", "mode", "time_signature"
)

model_df_srq1 <- song_level %>%
  dplyr::select(hit, all_of(audio_features))


# Exploratory Data Analysis (EDA)
set.seed(123)


# Boxplots of audio features
model_df_srq1 %>%
  dplyr::select(all_of(audio_features)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = feature, y = value)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Outlier Check via Boxplots")

key_audio_features <- c(
  "danceability", "energy", "valence", "tempo", "loudness",
  "acousticness", "instrumentalness", "speechiness", "liveness"
)

model_df_srq1 %>%
  group_by(hit) %>%
  summarise(across(all_of(key_audio_features), \(x) mean(x, na.rm = TRUE)))

model_df_srq1 %>%
  group_by(hit) %>%
  summarise(across(all_of(key_audio_features), \(x) sd(x, na.rm = TRUE)))

# Summary statistics for audio feature
summary(model_df_srq1[audio_features])
# Audio features distribution
model_df_srq1 %>%
  dplyr::select(all_of(key_audio_features)) %>%
  pivot_longer(everything(),
               names_to = "feature",
               values_to = "value") %>%
  drop_na(value) %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "darkorchid", alpha = 0.7) +
  facet_wrap(~ feature, scales = "free") +
  labs(title = "Distribution of Key Audio Features")


# Correlation between audio Features and response variable
pb_df <- model_df_srq1 %>%
  mutate(hit_num = as.numeric(hit))   
# Compute point-biserial correlations
pb_cor <- sapply(audio_features, function(var) {
  cor(pb_df[[var]], pb_df$hit_num, method = "pearson")
})
names(pb_cor) <- audio_features
par(mar = c(10, 4, 4, 2))  
barplot(pb_cor,
        las = 2,
        main = "Point-Biserial Correlation with Hit",
        ylab = "Correlation")


model_df_srq1_scaled <- model_df_srq1 %>%
  mutate(across(all_of(audio_features), scale))
# Fit Logistic Regression for AUC
logit_model <- glm(hit ~ ., data = model_df_srq1_scaled, family = "binomial")
summary(logit_model)
pred_prob <- predict(logit_model, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
mean(pred_class == model_df_srq1_scaled$hit)
roc_obj <- roc(model_df_srq1_scaled$hit, pred_prob)
auc(roc_obj)
coef_df <- tidy(logit_model) %>%
  filter(term != "(Intercept)")
ggplot(coef_df, aes(y = reorder(term, estimate), x = estimate)) +
  geom_point(size = 2) +
  geom_errorbar(
    aes(
      xmin = estimate - 1.96 * std.error,
      xmax = estimate + 1.96 * std.error
    ),
    width = 0.2,
    orientation = "y"   
  ) +
  labs(
    title = "Logistic Regression Coefficient Plot",
    x = "Coefficient Estimate",
    y = "Audio Feature"
  ) +
  theme_minimal()

pscl::pR2(logit_model)
# Variance Inflation Factors (VIF)
car::vif(logit_model)


#Correlation amongst audio features
png("qgraph_plot.png", width = 2000, height = 2000, res = 300)
cor_matrix <- cor(model_df_srq1 %>% dplyr::select(-hit))
qgraph(cor_matrix,
       layout = "spring",
       labels = colnames(cor_matrix),
       node.width = 2)

dev.off()

# SRQ2 - Train v/s test for models fitting
model_df_srq1$hit <- as.numeric(model_df_srq1$hit)
dummy_vars <- dummyVars(hit ~ ., data = model_df_srq1)
X_numeric <- predict(dummy_vars, newdata = model_df_srq1)
model_numeric <- data.frame(hit = model_df_srq1$hit, X_numeric)

# train test split
set.seed(123)
train_index <- createDataPartition(model_numeric$hit, p = 0.8, list = FALSE)
train_df <- model_numeric[train_index, ]
test_df  <- model_numeric[-train_index, ]
train_df$hit_factor <- factor(train_df$hit, levels = c(0,1), labels = c("No","Yes"))
test_df$hit_factor  <- factor(test_df$hit,  levels = c(0,1), labels = c("No","Yes"))
train_df$hit <- NULL
test_df$hit  <- NULL

# Scaling for KNN
predictor_cols <- setdiff(names(train_df), "hit_factor")
preProcValues <- preProcess(train_df[, predictor_cols], method = c("center","scale"))
train_scaled <- predict(preProcValues, train_df[, predictor_cols])
test_scaled  <- predict(preProcValues,  test_df[, predictor_cols])
train_scaled$hit_factor <- train_df$hit_factor
test_scaled$hit_factor  <- test_df$hit_factor

# Model fitting
# Logistic Regression
log_model <- glm(hit_factor ~ ., data = train_df, family = binomial)
log_train_pred <- ifelse(predict(log_model, type="response") > 0.5, "Yes", "No")
log_test_pred  <- ifelse(predict(log_model, newdata=test_df, type="response") > 0.5, "Yes", "No")
# LDA
lda_model <- lda(hit_factor ~ ., data=train_df)
lda_train_pred <- predict(lda_model)$class
lda_test_pred  <- predict(lda_model, newdata=test_df)$class
# QDA
qda_model <- qda(hit_factor ~ ., data=train_df)
qda_train_pred <- predict(qda_model)$class
qda_test_pred  <- predict(qda_model, newdata=test_df)$class
# KNN 
set.seed(123)
knn_control <- trainControl(
  method="cv",
  number=5,
  classProbs=TRUE,
  summaryFunction=twoClassSummary
)
k_grid <- expand.grid(k = seq(1, floor(sqrt(nrow(train_scaled))), by=2))

knn_tuned <- train(
  hit_factor ~ .,
  data=train_scaled,
  method="knn",
  tuneGrid=k_grid,
  trControl=knn_control,
  metric="ROC"
)
optimal_k <- knn_tuned$bestTune$k
knn_train_pred <- predict(knn_tuned, train_scaled)
knn_test_pred  <- predict(knn_tuned, test_scaled)

# Random Forest
rf_train_x <- train_df
rf_train_x$hit <- NULL
rf_train_x$hit_factor <- NULL

rf_test_x <- test_df
rf_test_x$hit <- NULL
rf_test_x$hit_factor <- NULL

set.seed(123)

rf_grid <- expand.grid(mtry = 1:6)
train_control <- trainControl(method = "cv", number = 5)

rf_cv <- train(
  x = rf_train_x,
  y = train_df$hit_factor,
  method = "rf",
  tuneGrid = rf_grid,
  trControl = train_control
)
optimal_mtry <- rf_cv$bestTune$mtry
optimal_mtry

rf_model <- randomForest(
  x = rf_train_x,
  y = train_df$hit_factor,
  mtry = optimal_mtry,
  nodesize = 10,
  maxnodes = 20
)

rf_train_pred <- predict(rf_model, rf_train_x)
rf_test_pred  <- predict(rf_model, rf_test_x)

rf_train_misclass <- mean(rf_train_pred != train_df$hit_factor)
rf_test_misclass  <- mean(rf_test_pred  != test_df$hit_factor)


# Gradient Boosting (GBM)
set.seed(123)
gbm_grid <- expand.grid(
  n.trees = c(100,200,300),
  interaction.depth = c(1,3),
  shrinkage = c(0.05,0.1),
  n.minobsinnode = 10
)
gbm_control <- trainControl(
  method="cv",
  number=5,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  savePredictions="final"
)

gbm_tuned <- train(
  hit_factor ~ .,
  data=train_df,
  method="gbm",
  tuneGrid=gbm_grid,
  trControl=gbm_control,
  metric="ROC",
  verbose=FALSE
)

optimal_trees   <- gbm_tuned$bestTune$n.trees
optimal_depth   <- gbm_tuned$bestTune$interaction.depth
optimal_shrink  <- gbm_tuned$bestTune$shrinkage
optimal_minobs  <- gbm_tuned$bestTune$n.minobsinnode
gbm_train_prob <- predict(gbm_tuned, train_df, type="prob")[, "Yes"]
gbm_test_prob  <- predict(gbm_tuned, test_df,  type="prob")[, "Yes"]
gbm_train_pred <- factor(ifelse(gbm_train_prob > 0.5, "Yes", "No"),
                         levels = c("No", "Yes"))

gbm_test_pred <- factor(ifelse(gbm_test_prob > 0.5, "Yes", "No"),
                        levels = c("No", "Yes"))
tuning_results <- data.frame(
  Method       = c("KNN", "Random Forest", "Gradient Boosting"),
  optimal_k    = c(optimal_k, NA, NA),
  optimal_mtry = c(NA, optimal_mtry, NA),
  n.trees      = c(NA, NA, optimal_trees),
  depth        = c(NA, NA, optimal_depth),
  shrinkage    = c(NA, NA, optimal_shrink),
  min_obs      = c(NA, NA, optimal_minobs)
)

tuning_results
tuning_results[is.na(tuning_results)] <- "–"
png("tuning_results.png", width = 1200, height = 600, res = 150)  # adjust size/res as needed
grid.table(tuning_results, rows = NULL)
dev.off()

# misclassification table
results <- data.frame(
  Model = c("Logistic", "LDA", "QDA", "KNN", "Random Forest", "GBM"),
  
  Train_Misclass = c(
    mean(log_train_pred != train_df$hit_factor),
    mean(lda_train_pred != train_df$hit_factor),
    mean(qda_train_pred != train_df$hit_factor),
    mean(knn_train_pred != train_df$hit_factor),
    mean(rf_train_pred  != train_df$hit_factor),
    mean(gbm_train_pred != train_df$hit_factor)
  ),
  
  Test_Misclass = c(
    mean(log_test_pred != test_df$hit_factor),
    mean(lda_test_pred != test_df$hit_factor),
    mean(qda_test_pred != test_df$hit_factor),
    mean(knn_test_pred != test_df$hit_factor),
    mean(rf_test_pred  != test_df$hit_factor),
    mean(gbm_test_pred != test_df$hit_factor)
  )
)

results
png("train_test_misclass_summarized.png", width = 1200, height = 600, res = 150)
grid.table(results, rows = NULL)
dev.off()   




# SRQ3 - Interaction terms preparation 
billboard$WeekID <- as.Date(billboard$WeekID, format = "%m/%d/%Y")
billboard_song <- billboard %>%
  group_by(SongID) %>%
  summarize(
    Performer = first(Performer),
    Song = first(Song),
    debut_week = min(WeekID),
    debut_position = min(Peak.Position),
    peak_position = min(Peak.Position),
    .groups = "drop"
  ) 
spotify_clean_srq3 <- spotify %>%
  dplyr::select(
    SongID,
    danceability, energy, key, loudness, mode, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo,
    time_signature, spotify_track_popularity,
    spotify_genre
  )
# Merge Billboard + Spotify using SongID only
merged1 <- billboard_song %>%
  left_join(spotify_clean_srq3, by = "SongID")

# Artist‑level stats from weekly Billboard data
artist_stats <- billboard %>%
  group_by(Performer) %>%
  summarize(
    artist_song_count = n_distinct(SongID),
    artist_total_weeks = n(),
    artist_avg_peak = mean(Peak.Position, na.rm = TRUE),
    .groups = "drop"
  )
merged2 <- merged1 %>%
  left_join(artist_stats, by = "Performer")

# primary genre from Spotify genre list
spotify_genre_clean <- spotify %>%
  mutate(
    primary_genre = case_when(
      spotify_genre == "[]" ~ "Unknown",
      TRUE ~ str_extract(spotify_genre, "(?<=\\[')([^']+)")
    )
  ) %>%
  dplyr::select(SongID, primary_genre)
genre_counts <- spotify_genre_clean %>%
  count(primary_genre)
common_genres <- genre_counts %>%
  filter(n >= 200) %>%
  pull(primary_genre)
spotify_genre_clean <- spotify_genre_clean %>%
  mutate(
    primary_genre_collapsed = ifelse(
      primary_genre %in% common_genres,
      primary_genre,
      "Other"
    )
  ) %>%
  group_by(SongID) %>%
  slice(1) %>%
  ungroup()

final_merged_df <- merged2 %>%
  left_join(
    spotify_genre_clean %>% dplyr::select(SongID, primary_genre_collapsed),
    by = "SongID"
  )

final_merged_df <- final_merged_df %>%
  distinct(SongID, .keep_all = TRUE) %>%
  drop_na()

names(final_merged_df)

final_merged_df <- final_merged_df %>%
  mutate(hit = factor(
    ifelse(peak_position <= 40, "Yes", "No"),
    levels = c("No", "Yes")
  ))

# Final dataset for SRQ3
srq3_df <- final_merged_df %>%
  dplyr::select(
    hit,
    danceability, energy, loudness, speechiness, acousticness,
    instrumentalness, liveness, valence, tempo,
    key, mode, time_signature,
    primary_genre_collapsed,
    artist_song_count, artist_total_weeks, artist_avg_peak
  )
names(srq3_df)

srq3_df$hit <- as.factor(srq3_df$hit)
srq3_df$primary_genre_collapsed <- factor(srq3_df$primary_genre_collapsed)
srq3_df$key <- factor(srq3_df$key)
srq3_df$mode <- factor(srq3_df$mode)
srq3_df$time_signature <- factor(srq3_df$time_signature)


# Audio × Audio interactions
srq3_df$energy_danceability <- srq3_df$energy * srq3_df$danceability
srq3_df$valence_tempo <- srq3_df$valence * srq3_df$tempo
#  Genre × Audio interactions
srq3_df$genre_energy <- interaction(
  srq3_df$primary_genre_collapsed,
  cut(srq3_df$energy, 3)
)
#  Artist × Audio interaction
srq3_df$artist_popularity_energy <- srq3_df$artist_avg_peak * srq3_df$energy

# Train/test split
set.seed(123)
train_index <- createDataPartition(srq3_df$hit, p = 0.8, list = FALSE)
train_srq3 <- srq3_df[train_index, ]
test_srq3  <- srq3_df[-train_index, ]

num_cols <- sapply(train_srq3, is.numeric)
preproc <- preProcess(train_srq3[, num_cols], method = c("center", "scale"))
train_srq3_scaled <- train_srq3
train_srq3_scaled[, num_cols] <- predict(preproc, train_srq3[, num_cols])
test_srq3_scaled <- test_srq3
test_srq3_scaled[, num_cols] <- predict(preproc, test_srq3[, num_cols])

# Define positive/negative class
pos_class <- levels(train_srq3$hit)[2]   # " hit = Yes"
neg_class <- levels(train_srq3$hit)[1]   # "hit = No"

# Logistic Regression
set.seed(123)
logit_srq3 <- glm(
  hit ~ ., 
  data = train_srq3,
  family = binomial
)
logit_srq3_train_prob <- predict(
  logit_srq3,
  type = "response"
)
logit_srq3_test_prob <- predict(
  logit_srq3,
  newdata = test_srq3,
  type = "response"
)
logit_srq3_train_pred <- factor(
  ifelse(logit_srq3_train_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)
logit_srq3_test_pred <- factor(
  ifelse(logit_srq3_test_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)
# LDA
set.seed(123)
lda_srq3 <- lda(hit ~ ., data = train_srq3)
lda_train_prob <- predict(lda_srq3)$posterior[, pos_class]
lda_test_prob  <- predict(lda_srq3, newdata = test_srq3)$posterior[, pos_class]
lda_train_pred <- factor(
  ifelse(lda_train_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)
lda_test_pred <- factor(
  ifelse(lda_test_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)

# Random Forest 
set.seed(123)
mtry_grid <- c(1, 2, 3, 4)
min_node_grid <- c(5, 10, 20)

rf_grid <- expand.grid(
  mtry = mtry_grid,
  min.node.size = min_node_grid
)

rf_grid$AUC <- NA 
for (i in 1:nrow(rf_grid)) {
  
  rf_model_tmp <- ranger(
    hit ~ .,
    data = train_srq3,
    probability = TRUE,
    num.trees = 500,
    mtry = rf_grid$mtry[i],
    min.node.size = rf_grid$min.node.size[i],
    max.depth = 10,
    sample.fraction = 0.7
  )
  
    tmp_prob <- predict(rf_model_tmp, test_srq3)$predictions[, pos_class]
    rf_grid$AUC[i] <- pROC::roc(test_srq3$hit, tmp_prob)$auc
}
best_row <- rf_grid[which.max(rf_grid$AUC), ]
best_row

rf_reg <- ranger(
  hit ~ .,
  data = train_srq3,
  probability = TRUE,
  num.trees = 500,
  mtry = best_row$mtry,
  min.node.size = best_row$min.node.size,
  max.depth = 10,
  sample.fraction = 0.7
)
rf_train_prob <- predict(rf_reg, train_srq3)$predictions[, pos_class]
rf_test_prob  <- predict(rf_reg, test_srq3)$predictions[, pos_class]

# Class predictions
rf_train_pred <- factor(
  ifelse(rf_train_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)

rf_test_pred <- factor(
  ifelse(rf_test_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)

# KNN(Scaled)
train_x <- train_srq3_scaled[, sapply(train_srq3_scaled, is.numeric)]
test_x  <- test_srq3_scaled[,  sapply(test_srq3_scaled, is.numeric)]

train_y <- train_srq3_scaled$hit
k_grid <- expand.grid(k = seq(1, floor(sqrt(nrow(train_x))), by = 2))
set.seed(123)
folds <- sample(rep(1:5, length.out = nrow(train_x)))

cv_errors <- numeric(length(k_grid))

for (i in seq_along(k_grid)) {
  
  k_val <- k_grid$k[i]
  fold_err <- numeric(5)
  
  for (f in 1:5) {
    idx_test  <- which(folds == f)
    idx_train <- which(folds != f)
    
    pred <- knn(
      train = train_x[idx_train, ],
      test  = train_x[idx_test, ],
      cl    = train_y[idx_train],
      k     = k_val
    )
    
    fold_err[f] <- mean(pred != train_y[idx_test])
  }
  
  cv_errors[i] <- mean(fold_err)
}
k_opt <- k_grid$k[which.min(cv_errors)]
k_opt
knn_train_pred_raw <- knn(
  train = train_x,
  test  = train_x,
  cl    = train_y,
  k     = k_opt,
  prob  = TRUE
)

knn_train_prob <- ifelse(
  knn_train_pred_raw == pos_class,
  attr(knn_train_pred_raw, "prob"),
  1 - attr(knn_train_pred_raw, "prob")
)

knn_train_pred <- factor(
  ifelse(knn_train_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)

knn_test_pred_raw <- knn(
  train = train_x,
  test  = test_x,
  cl    = train_y,
  k     = k_opt,
  prob  = TRUE
)

knn_test_prob <- ifelse(
  knn_test_pred_raw == pos_class,
  attr(knn_test_pred_raw, "prob"),
  1 - attr(knn_test_pred_raw, "prob")
)

knn_test_pred <- factor(
  ifelse(knn_test_prob > 0.5, pos_class, neg_class),
  levels = c(neg_class, pos_class)
)

# GBM 
set.seed(123)
train_srq3$hit <- factor(train_srq3$hit, levels = c("No", "Yes"))
test_srq3$hit  <- factor(test_srq3$hit,  levels = c("No", "Yes"))
pos_class <- "Yes"
neg_class <- "No"
grid <- expand.grid(
  n.trees = c(200, 400, 600),
  interaction.depth = c(1, 2),
  shrinkage = c(0.01, 0.05),
  n.minobsinnode = 5
)

grid$AUC <- NA
train_srq3_gbm <- train_srq3
test_srq3_gbm  <- test_srq3
train_srq3_gbm$hit <- ifelse(train_srq3_gbm$hit == "Yes", 1, 0)
test_srq3_gbm$hit  <- ifelse(test_srq3_gbm$hit == "Yes", 1, 0)
for (i in 1:nrow(grid)) {
  
  gbm_tmp <- gbm(
    formula = hit ~ .,
    data = train_srq3_gbm,
    distribution = "bernoulli",
    n.trees = grid$n.trees[i],
    interaction.depth = grid$interaction.depth[i],
    shrinkage = grid$shrinkage[i],
    n.minobsinnode = grid$n.minobsinnode[i],
    bag.fraction = 0.7,
    train.fraction = 1.0,
    verbose = FALSE
  )
 tmp_prob <- predict(gbm_tmp, test_srq3_gbm, n.trees = grid$n.trees[i], type = "response")
  
  grid$AUC[i] <- pROC::roc(
    test_srq3_gbm$hit,
    tmp_prob,
    quiet = TRUE
  )$auc
}

best_row <- grid[which.max(grid$AUC), ]
best_row

gbm_final <- gbm(
  hit ~ .,
  data = train_srq3_gbm,
  distribution = "bernoulli",
  n.trees = best_row$n.trees,
  interaction.depth = best_row$interaction.depth,
  shrinkage = best_row$shrinkage,
  n.minobsinnode = best_row$n.minobsinnode,
  bag.fraction = 0.7,
  train.fraction = 1.0,
  verbose = FALSE
)
gbm_train_prob <- predict(gbm_final, train_srq3_gbm,
                          n.trees = best_row$n.trees,
                          type = "response")

gbm_test_prob <- predict(gbm_final, test_srq3_gbm,
                         n.trees = best_row$n.trees,
                         type = "response")
gbm_train_pred <- factor(ifelse(gbm_train_prob > 0.5, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
gbm_test_pred <- factor(ifelse(gbm_test_prob > 0.5, pos_class, neg_class),
                        levels = c(neg_class, pos_class))

# SHAP Analysis for SRQ4 (Evaluation part) 

set.seed(321)
shap_pred_rf <- function(model, newdata) {
  predict(model, data = newdata)$predictions[, pos_class]
}

n_shap <- min(120L, nrow(test_srq3))
shap_idx <- sample.int(nrow(test_srq3), n_shap)

feat_names <- setdiff(names(test_srq3), "hit")
X_shap_rf <- test_srq3[shap_idx, feat_names, drop = FALSE]

shap_rf_mat <- fastshap::explain(
  rf_reg,
  X = X_shap_rf,
  pred_wrapper = shap_pred_rf,
  nsim = 50,     
  adjust = TRUE
)

rf_shap_importance <- data.frame(
  Feature = colnames(shap_rf_mat),
  MeanAbsSHAP = apply(abs(shap_rf_mat), 2, mean)
)

rf_shap_importance <- rf_shap_importance[order(-rf_shap_importance$MeanAbsSHAP), ]
rf_shap_importance

shap_pred_gbm <- function(model, newdata) {
  predict(model, newdata,
          n.trees = best_row$n.trees,
          type = "response")
}

n_shap <- min(120L, nrow(test_srq3))
shap_idx <- sample.int(nrow(test_srq3), n_shap)

feat_names <- setdiff(names(test_srq3), "hit")
X_shap_gbm <- test_srq3[shap_idx, feat_names, drop = FALSE]

shap_gbm_mat <- fastshap::explain(
  gbm_final,
  X = X_shap_gbm,
  pred_wrapper = shap_pred_gbm,
  nsim = 50,
  adjust = TRUE
)

gbm_shap_importance <- data.frame(
  Feature = colnames(shap_gbm_mat),
  MeanAbsSHAP = apply(abs(shap_gbm_mat), 2, mean)
)

gbm_shap_importance <- gbm_shap_importance[order(-gbm_shap_importance$MeanAbsSHAP), ]
gbm_shap_importance

rf_imp_df <- data.frame(
  Feature = colnames(shap_rf_mat),
  MeanAbsSHAP = apply(abs(shap_rf_mat), 2, mean)
)
rf_imp_df <- rf_imp_df[order(-rf_imp_df$MeanAbsSHAP), ]

png(file.path(getwd(), "shap_rf_importance.png"), width = 1600, height = 900, res = 160)
print(
  ggplot(rf_imp_df, aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest SHAP Importance",
         x = "Feature",
         y = "Mean |SHAP|") +
    theme_minimal(base_size = 16)
)
dev.off()

gbm_imp_df <- data.frame(
  Feature = colnames(shap_gbm_mat),
  MeanAbsSHAP = apply(abs(shap_gbm_mat), 2, mean)
)
gbm_imp_df <- gbm_imp_df[order(-gbm_imp_df$MeanAbsSHAP), ]

png(file.path(getwd(), "shap_gbm_importance.png"), width = 1600, height = 900, res = 160)
print(
  ggplot(gbm_imp_df, aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
    geom_col(fill = "darkgreen") +
    coord_flip() +
    labs(title = "GBM SHAP Importance",
         x = "Feature",
         y = "Mean |SHAP|") +
    theme_minimal(base_size = 16)
)
dev.off()



# Evaluation
# Confusion matrices on the test set (positive class = hit "1")
# caret: rows = what we predicted, columns = actual (reference)
cm_logit_test <- confusionMatrix(logit_srq3_test_pred, test_srq3$hit, positive = pos_class)
cm_lda_test   <- confusionMatrix(lda_test_pred, test_srq3$hit, positive = pos_class)
cm_knn_test   <- confusionMatrix(knn_test_pred, test_srq3_scaled$hit, positive = pos_class)
cm_rf_test    <- confusionMatrix(rf_test_pred, test_srq3$hit, positive = pos_class)
cm_gbm_test   <- confusionMatrix(gbm_test_pred, test_srq3$hit, positive = pos_class)

# SRQ3 Misclassification Table + test accuracy, precision, recall, AUC
srq3_results <- data.frame(
  Model = c("Logistic", "LDA", "KNN", "Random Forest", "GBM"),
  Train_Misclass = c(
    mean(logit_srq3_train_pred != train_srq3$hit),
    mean(lda_train_pred != train_srq3$hit),
    mean(knn_train_pred != train_y),
    mean(rf_train_pred != train_srq3$hit),
    mean(gbm_train_pred != train_srq3$hit)
  ),
  Test_Misclass = c(
    mean(logit_srq3_test_pred != test_srq3$hit),
    mean(lda_test_pred != test_srq3$hit),
    mean(knn_test_pred != test_srq3_scaled$hit),
    mean(rf_test_pred != test_srq3$hit),
    mean(gbm_test_pred != test_srq3$hit)
  ),
  Test_Accuracy = c(
    cm_logit_test$overall["Accuracy"],
    cm_lda_test$overall["Accuracy"],
    cm_knn_test$overall["Accuracy"],
    cm_rf_test$overall["Accuracy"],
    cm_gbm_test$overall["Accuracy"]
  ),
  Test_Precision = c(
    cm_logit_test$byClass["Pos Pred Value"],
    cm_lda_test$byClass["Pos Pred Value"],
    cm_knn_test$byClass["Pos Pred Value"],
    cm_rf_test$byClass["Pos Pred Value"],
    cm_gbm_test$byClass["Pos Pred Value"]
  ),
  Test_Recall = c(
    cm_logit_test$byClass["Sensitivity"],
    cm_lda_test$byClass["Sensitivity"],
    cm_knn_test$byClass["Sensitivity"],
    cm_rf_test$byClass["Sensitivity"],
    cm_gbm_test$byClass["Sensitivity"]
  ),
  Test_AUC = c(
    auc(roc(test_srq3$hit, logit_srq3_test_prob)),
    auc(roc(test_srq3$hit, lda_test_prob)),
    auc(roc(test_srq3_scaled$hit, knn_test_prob)),
    auc(roc(test_srq3$hit, rf_test_prob)),
    auc(roc(test_srq3$hit, gbm_test_prob))
  )
)

srq3_results
srq3_results_perf <- srq3_results[, c(
  "Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_AUC"
)]
srq3_results_errors <- srq3_results[, c("Model", "Train_Misclass", "Test_Misclass")]

round_numeric_df <- function(df) {
  df[, sapply(df, is.numeric)] <- round(df[, sapply(df, is.numeric)], 4)
  df
}
srq3_results_perf_print <- round_numeric_df(srq3_results_perf)
srq3_results_errors_print <- round_numeric_df(srq3_results_errors)

srq3_table_theme <- ttheme_default(
  core = list(fg_params = list(cex = 1.15)),
  colhead = list(fg_params = list(cex = 1.2, fontface = "bold"))
)
png("train_test_performance_srq3.png", width = 1600, height = 550, res = 160)
grid.table(srq3_results_perf_print, rows = NULL, theme = srq3_table_theme)
dev.off()
png("train_test_errors_srq3.png", width = 1600, height = 550, res = 160)
grid.table(srq3_results_errors_print, rows = NULL, theme = srq3_table_theme)
dev.off()

# Paired t-tests: Random Forest as best model vs other models
y_test <- test_srq3$hit
correct_rf <- as.integer(rf_test_pred == y_test)
correct_logit <- as.integer(logit_srq3_test_pred == y_test)
correct_lda <- as.integer(lda_test_pred == y_test)
correct_knn <- as.integer(knn_test_pred == test_srq3_scaled$hit)
correct_gbm <- as.integer(gbm_test_pred == y_test)

cat("\n--- SRQ3 paired t-tests: Random Forest vs other models (test set) ---\n")

tt_rf_vs_logit <- t.test(correct_rf, correct_logit, paired = TRUE)
tt_rf_vs_lda   <- t.test(correct_rf, correct_lda, paired = TRUE)
tt_rf_vs_knn   <- t.test(correct_rf, correct_knn, paired = TRUE)
tt_rf_vs_gbm   <- t.test(correct_rf, correct_gbm, paired = TRUE)

print(tt_rf_vs_logit)
print(tt_rf_vs_lda)
print(tt_rf_vs_knn)
print(tt_rf_vs_gbm)

srq3_rf_paired_t <- data.frame(
  Comparison = c("RF vs Logistic", "RF vs LDA", "RF vs KNN", "RF vs GBM"),
  Mean_accuracy_diff = c(
    mean(correct_rf - correct_logit),
    mean(correct_rf - correct_lda),
    mean(correct_rf - correct_knn),
    mean(correct_rf - correct_gbm)
  ),
  p_value = c(
    tt_rf_vs_logit$p.value,
    tt_rf_vs_lda$p.value,
    tt_rf_vs_knn$p.value,
    tt_rf_vs_gbm$p.value
  )
)
srq3_rf_paired_t$p_value <- round(srq3_rf_paired_t$p_value, 5)
srq3_rf_paired_t$Mean_accuracy_diff <- round(srq3_rf_paired_t$Mean_accuracy_diff, 5)

cat("Comparison", "Mean_accuracy_diff", "  ", "p_value", "\n")

for (i in 1:nrow(srq3_rf_paired_t)) {
  cat(
    srq3_rf_paired_t$Comparison[i],
    srq3_rf_paired_t$Mean_accuracy_diff[i],
    "  ",   
    srq3_rf_paired_t$p_value[i],
    "\n"
  )
}
print(srq3_rf_paired_t, right = FALSE)



# Performed PCA for SRQ4
set.seed(123)
srq4_df <- song_level %>%
  dplyr::select(hit, all_of(audio_features))
srq4_df$hit <- as.factor(srq4_df$hit)
# numeric predictors for PCA
srq4_numeric <- srq4_df %>%
  dplyr::select(where(is.numeric))
srq4_scaled <- scale(srq4_numeric)

pca_model <- prcomp(srq4_scaled, center = TRUE, scale. = TRUE)

# Determine principal components ~90% variance
var_explained <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
num_pc <- which(var_explained >= 0.90)[1]
cat("Number of PCs for 90% variance:", num_pc, "\n")
X_pcs <- as.data.frame(pca_model$x[, 1:num_pc])
X_pcs$hit <- srq4_df$hit

# Train/Test Split
train_index <- createDataPartition(X_pcs$hit, p = 0.8, list = FALSE)
train_pca <- X_pcs[train_index, ]
test_pca  <- X_pcs[-train_index, ]
pos_class <- levels(srq4_df$hit)[2]
neg_class <- levels(srq4_df$hit)[1]

# Logistic Regression 
logit_pca <- glm(hit ~ ., data = train_pca, family = binomial)
logit_train_prob <- predict(logit_pca, type = "response")
logit_train_pred <- factor(ifelse(logit_train_prob > 0.5, pos_class, neg_class),
                           levels = c(neg_class, pos_class))
logit_test_prob <- predict(logit_pca, newdata = test_pca, type = "response")
logit_test_pred <- factor(ifelse(logit_test_prob > 0.5, pos_class, neg_class),
                          levels = c(neg_class, pos_class))

# LDA on PCs
lda_pca <- lda(hit ~ ., data = train_pca)
lda_train_prob <- predict(lda_pca)$posterior[, pos_class]
lda_test_prob  <- predict(lda_pca, newdata = test_pca)$posterior[, pos_class]

lda_train_pred <- factor(ifelse(lda_train_prob > 0.5, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
lda_test_pred <- factor(ifelse(lda_test_prob > 0.5, pos_class, neg_class),
                        levels = c(neg_class, pos_class))

#  QDA on PCs
qda_pca <- qda(hit ~ ., data = train_pca)
qda_train_prob <- predict(qda_pca)$posterior[, pos_class]
qda_test_prob  <- predict(qda_pca, newdata = test_pca)$posterior[, pos_class]

qda_train_pred <- factor(ifelse(qda_train_prob > 0.5, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
qda_test_pred <- factor(ifelse(qda_test_prob > 0.5, pos_class, neg_class),
                        levels = c(neg_class, pos_class))

# 8.4 KNN on PCs with tuning
train_knn_x <- train_pca %>% dplyr::select(-hit)
test_knn_x  <- test_pca %>% dplyr::select(-hit)
train_knn_y <- train_pca$hit

# 5-fold CV tuning for k
folds <- sample(rep(1:5, length.out = nrow(train_knn_x)))
k_values <- seq(1, floor(sqrt(nrow(train_knn_x))), by = 2)
cv_errors <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k_val <- k_values[i]
  fold_err <- numeric(5)
  for (f in 1:5) {
    idx_test  <- which(folds == f)
    idx_train <- which(folds != f)
    pred <- knn(train = train_knn_x[idx_train, ],
                test  = train_knn_x[idx_test, ],
                cl    = train_knn_y[idx_train],
                k     = k_val)
    fold_err[f] <- mean(pred != train_knn_y[idx_test])
  }
  cv_errors[i] <- mean(fold_err)
}
# Best k
k_opt <- k_values[which.min(cv_errors)]
cat("Optimal k for KNN on PCs:", k_opt, "\n")
# Final KNN predictions
knn_train_pred_raw <- knn(train_knn_x, train_knn_x, train_knn_y, k = k_opt, prob = TRUE)
knn_test_pred_raw  <- knn(train_knn_x, test_knn_x, train_knn_y, k = k_opt, prob = TRUE)

knn_train_prob <- ifelse(knn_train_pred_raw == pos_class,
                         attr(knn_train_pred_raw, "prob"),
                         1 - attr(knn_train_pred_raw, "prob"))
knn_test_prob <- ifelse(knn_test_pred_raw == pos_class,
                        attr(knn_test_pred_raw, "prob"),
                        1 - attr(knn_test_pred_raw, "prob"))

knn_train_pred <- factor(ifelse(knn_train_prob > 0.5, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
knn_test_pred  <- factor(ifelse(knn_test_prob > 0.5, pos_class, neg_class),
                         levels = c(neg_class, pos_class))

#  Model results of Misclassification & AUC
srq4_results <- data.frame(
  Model = c("Logistic (PCA)", "LDA (PCA)", "QDA (PCA)", "KNN (PCA)"),
  
  Train_Misclass = c(
    mean(logit_train_pred != train_pca$hit),
    mean(lda_train_pred != train_pca$hit),
    mean(qda_train_pred != train_pca$hit),
    mean(knn_train_pred != train_pca$hit)
  ),
  
  Test_Misclass = c(
    mean(logit_test_pred != test_pca$hit),
    mean(lda_test_pred != test_pca$hit),
    mean(qda_test_pred != test_pca$hit),
    mean(knn_test_pred != test_pca$hit)
  ),
  
  Test_AUC = c(
    auc(roc(test_pca$hit, logit_test_prob)),
    auc(roc(test_pca$hit, lda_test_prob)),
    auc(roc(test_pca$hit, qda_test_prob)),
    auc(roc(test_pca$hit, knn_test_prob))
  )
)
srq4_results
png("train_test_misclass_srq4.png", width = 1200, height = 600, res = 150)
grid.table(srq4_results, rows = NULL)
dev.off()

