# Advertising Sales Prediction

This project demonstrates a complete workflow for predicting sales based on advertising spends using a Linear Regression model.

## Overview

The goal of this project is to predict sales based on the amount spent on different advertising channels: TV, Radio, and Newspaper. The project uses a Linear Regression model for the prediction task.

## Dataset

The dataset used in this project contains advertising data with the following columns:
- `TV`: Advertising spend on TV.
- `Radio`: Advertising spend on Radio.
- `Newspaper`: Advertising spend on Newspapers.
- `Sales`: Sales generated.

## Project Structure

- `advertising_sales_prediction.ipynb`: Jupyter notebook containing the entire workflow from data loading, preprocessing, model training, evaluation, and visualization.
- `assets/`: Directory containing images used in the README.

## Model Description

The model used in this project is a Linear Regression model, which is a simple yet powerful regression technique.

1. **Data Preprocessing**:
   - **Null Values Check**: Checked for any null values in the dataset.
   - **Correlation Analysis**: Analyzed the correlation between features and the target variable (Sales).

2. **Linear Regression Model**:
   - **Purpose**: The Linear Regression model is used to predict sales based on advertising spends on TV, Radio, and Newspaper.
   - **Training**: The model is trained on 80% of the dataset and tested on the remaining 20%.

## Data Visualization

### Predicted vs Actual Values

![Predicted vs Actual Values](assets/predicted%20vs%20actual%20values.png)

### Residuals Plot

![Residuals Plot](assets/Residual%20plot.png)

### Residuals Histogram

![Residuals Histogram](assets/Residual%20histogram.png)
