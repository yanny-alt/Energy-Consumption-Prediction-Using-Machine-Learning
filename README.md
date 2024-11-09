# Predicting Daily Power Consumption: A Supervised Regression Approach
![](https://github.com/yanny-alt/Energy-Consumption-Prediction-Using-Machine-Learning/blob/main/images/imagegeneratedbyai.png)

## 1. Introduction
This project explores daily power consumption patterns over several years to predict future usage accurately. Using machine learning models, we aim to assist energy providers in balancing energy distribution, optimizing usage, and forecasting demand. Our supervised regression approach evaluates various models, focusing on the prediction accuracy of future daily power consumption.

## 2. Project Overview
- **Project Type**: Supervised Regression (Time Series Prediction)
- **Dataset**: Power Consumption Dataset from UCI Machine Learning Repository
- **Goal**: Predict daily power consumption and evaluate model performance
- **Techniques**: Random Forest, XGBoost, and other regression algorithms
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

## 3. Problem Statement
Accurate power consumption prediction helps energy providers optimize energy generation and distribution, reducing waste. Due to the variability in daily consumption influenced by multiple factors, building robust predictive models is essential for reliable energy forecasting.

## 4. Project Objectives
- **Build a Predictive Model**: Use historical data to forecast daily power consumption.
- **Compare Models**: Evaluate the performance of different supervised regression models.
- **Optimize RMSE**: Ensure the best model achieves an RMSE below 450 kW.
- **Analyze Trends**: Compare predicted consumption trends with actual data.

## 5. Skills Demonstrated / Technologies Used
- **Data Analysis & Visualization**: Using Python libraries such as `matplotlib` for visual insights
- **Machine Learning**: Random Forest, XGBoost, and model optimization
- **Model Evaluation**: RMSE as the primary metric
- **Documentation**: Clear documentation using Jupyter Notebooks and GitHub

## 6. Dataset Overview
The dataset includes daily power consumption over several years. Each entry records:
- **Date**: Date of measurement
- **Power Consumption**: Daily usage in kW
- **Other Features**: Year, semester, quarter, day of the week, week of the year, and month

The dataset is provided in two files:
- `df_train.csv` (Training Data)
- `df_test.csv` (Testing Data)

[Dataset Source Link](https://github.com/yanny-alt/Energy-Consumption-Prediction-Using-Machine-Learning/tree/main/datasets)

## 7. Project Methodology
### Data Preprocessing
1. **Data Inspection**: Load and review the dataset.
2. **Missing Values**: Check and handle missing values.
3. **Feature Engineering**: Convert categorical variables to numerical formats.
4. **Normalization**: Standardize data where needed.

### Model Selection
1. **Random Forest Regression**: To leverage feature importance in predictions.
2. **XGBoost**: Known for effective boosting and generalization.

### Model Training & Evaluation
1. **Training**: Models were trained on the training set.
2. **Evaluation Metric**: RMSE used to assess performance.
3. **Optimization**: Hyperparameters were tuned to reduce RMSE below 450 kW.

### Trend Analysis
1. **Visualization**: Plots of actual vs. predicted values were generated.
2. **Trend Similarity**: Visual checks confirmed poor prediction accuracy in following consumption trends.

## 8. Results

- **Best Model**: Random Forest with an RMSE of 435.02 kW.
- **Trend Analysis**: Predictions didn't follow similar patterns to actual data, demonstrating poor reliable model performance.

![Actual vs Predicted Power Consumption](https://github.com/yanny-alt/Energy-Consumption-Prediction-Using-Machine-Learning/blob/main/images/Predicted%20vs%20Actual%20Daily%20Power%20Consumption%20Plot%20Image.png)

## 9. Conclusion
This project explored predicting daily power consumption using supervised regression models. By leveraging historical consumption data, we aimed to build an accurate model for energy forecasting. The Random Forest model achieved an RMSE of 435.02 kW, indicating a relatively low prediction error.

However, upon analyzing the trend similarity between predicted and actual values, the correlation coefficient was below 0.9, leading to a "No" result for trend similarity. This outcome suggests that while the model's predictions are reasonably close in terms of error, they do not follow the exact trend of actual consumption. This discrepancy may be due to the lack of external factors such as weather or seasonal events in the dataset, which could improve the model's ability to capture fluctuations in consumption patterns.

Overall, the project demonstrates the potential of machine learning in energy forecasting but also highlights areas for further improvement, particularly in incorporating additional external variables to enhance trend alignment and predictive accuracy.

## 10. Future Recommendations
- **Include External Variables**: Integrate weather, holiday, and household data for more accurate predictions.
- **Refine Models**: Test neural networks or advanced time series models for further improvement.
- **Advanced Time Series Analysis**: Implement models like ARIMA or LSTM to capture complex trends.

