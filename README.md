# Predicting Positive Reviews in E-Commerce

## Project Overview
This project builds a predictive model using e-commerce data to identify customers likely to leave positive reviews. The CRISP-DM methodology was followed, covering business understanding, data preparation, modeling, and evaluation. The best-performing model, CatBoost, was deployed using a Streamlit-based prototype.

## Dataset
The dataset, sourced from Olist (Brazilian e-commerce platform), consists of 100,000 orders (2016–2018) and includes:
- Orders (purchase, payment, delivery details)
- Products & Categories
- Customers & Sellers
- Payments & Reviews

## Data Processing & Feature Engineering
- Merging multiple datasets via inner joins
- Creating derived features (e.g., product volume, delivery time, review verbosity)
- Encoding categorical variables & scaling numerical features
- Addressing class imbalance with SMOTE
- Dropping non-predictive features

## Model Selection & Evaluation
Trained models:
- **CatBoost** (Best-performing model)
- XGBoost, LightGBM, Gradient Boosting, Random Forest

| Model        | Test Accuracy | Precision | Recall | F1-Score |
|-------------|--------------|-----------|--------|---------|
| **CatBoost**  | **88.47%**   | **0.86**   | **0.95**  | **0.90**  |
| Stacked Model | 88.56%      | 0.89      | 0.94   | 0.90   |

### Why CatBoost?
- Strong generalization with minimal overfitting
- High recall and balanced F1-score
- Computational efficiency over complex stacked models

## Deployment
A Streamlit-based web application was developed to provide a user-friendly interface for predicting review sentiment. Users can input customer and order details to get predictions.

## Future Work
- Integrate AI and cloud computing for scalability
- Implement a change management plan for business adoption
- Enhance predictive capabilities with additional features

## Conclusion
This project delivers a robust predictive tool for Nile eCommerce to optimize marketing and enhance customer engagement. The CatBoost model is recommended for deployment based on its superior performance and efficiency.

