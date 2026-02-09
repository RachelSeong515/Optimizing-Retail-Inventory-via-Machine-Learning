# Optimizing-Retail-Inventory-via-Machine-Learning

# Project Description

Walmart operates in a highly competitive retail environment where accurate demand forecasting is essential for maintaining operational efficiency across thousands of stores nationwide. Inaccurate forecasts expose the business to significant risks, including lost revenue from stockouts during peak demand and higher costs from excess inventory during economic slowdowns.

The objective of this project is to develop reliable weekly sales forecasts for Walmart stores across the U.S. to support dynamic, data-driven inventory management.


# EDA (Data Description)

The analysis uses panel data from 45 Walmart stores over 143 weeks, totaling 6,435 observations. Key variables include weekly sales, holiday indicators, temperature, fuel price, CPI, and unemployment.

1. Weekly Sales vs. Temperature
Holiday sales consistently exceed non-holiday sales across all temperatures. Sales peak around 50°F, where the gap between holiday and non-holiday sales narrows, potentially reflecting increased in-store shopping under moderate weather conditions. At warmer temperatures (around 80°F), the curves converge, suggesting strong in-store demand regardless of holiday status.

2. Sales Density Distribution
Both holiday and non-holiday sales peak near $1M per week, the most common sales level. Holiday sales shift rightward, confirming higher overall demand, but exhibit greater variance. Non-holiday sales are more stable and predictable.

3. Weekly Sales vs. Unemployment
The relationship between unemployment and sales is non-linear, making linear regression insufficient on its own. Tree-based models (Random Forest, CART) better capture this behavior. Most observations fall in the 6–9% unemployment range, with large store-level variation, indicating unemployment alone does not explain sales outcomes.


# Data Preparation

1. Panel Data Treatment

The dataset was treated as panel data, with store IDs modeled as factor variables to capture persistent store-level effects while leveraging cross-store information.

2. Data Cleaning & Validation

No missing values or duplicates. Validated date parsing, value ranges, and store–date uniqueness. Checked distributions and correlations (moderate CPI–unemployment collinearity)

3. Train–Test Split

To preserve temporal integrity and prevent leakage, the first 80% of weeks were used for training and the remaining 20% for testing, ensuring evaluation on unseen future periods.

# Model

Weekly sales were forecast using temperature, fuel price, CPI, unemployment, and holiday indicators. Five models were built and compared: Linear Regression with Store Fixed Effects, CART (Decision Tree), Random Forest, Lasso, Post-Lasso

All models were trained using 10-fold cross-validation to ensure stability and reduce overfitting. This approach allowed comparison of linear and non-linear methods while balancing interpretability and predictive accuracy.
All models were trained using 10-fold cross-validation to ensure stability and reduce overfitting. This mix of linear and non-linear approaches allowed us to balance interpretability, model complexity, and predictive performance.


# Model Results & Findings

1. Random Forest - RMSE(161,153) - Best overall performance
2. Linear Regression - RMSE(173,353) - Baseline model
3. CART - RMSE(189,764) - Captures non-linearity, less stable
4. Lasso - RMSE(173,404) - Feature selection
5. Post-Lasso - RMSE(173,353) - Reduced bias


# Business Implications

The final Random Forest model is deployed as a production-ready forecasting tool to support weekly inventory planning across Walmart stores in the U.S. It generates store-level sales forecasts using recent economic and seasonal data, enabling more accurate replenishment decisions.

The system follows a monthly rolling retraining schedule to adapt to changing consumer behavior and macroeconomic conditions. Forecast accuracy is continuously monitored using out-of-sample RMSE, with alerts triggered when predictions fall outside predefined confidence bands.

Outputs are delivered in a simple, actionable format—a weekly forecast table containing store ID, week, and predicted sales—allowing seamless integration into existing supply chain and inventory workflows.

By reducing forecast error and improving demand visibility, this deployment helps lower stockout risk, reduce excess inventory, and support data-driven decision-making across retail operations.
