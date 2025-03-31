import streamlit as st
import joblib
import pandas as pd


model = joblib.load('cat_model.pkl')  

st.title("Nile eCommerce Review Prediction App")
st.write("Predict whether a customer will leave a positive review.")

# Numeric Input Fields
price = st.number_input("Price ($)", min_value=0.0, format="%.2f")
freight_value = st.number_input("Freight Value ($)", min_value=0.0, format="%.2f")
product_weight_g = st.number_input("Product Weight (grams)", min_value=0)
payment_value = st.number_input("Payment Value ($)", min_value=0.0, format="%.2f")
product_volume = st.number_input("Product Volume (cm³)", min_value=0)
delivery_duration_days = st.number_input("Delivery Duration (days)", min_value=0)
estimated_delivery_days = st.number_input("Estimated Delivery Days", min_value=0)
ontime_delivery = st.number_input("On-time Delivery (1 for yes, 0 for no)", min_value=0, max_value=1)
order_delay = st.number_input("Order Delay (days)", min_value=0)
review_length = st.number_input("Review Length (characters)", min_value=0)
customer_total_orders = st.number_input("Total Orders by Customer", min_value=0)

# Dropdowns for Categorical Data
customer_state = st.selectbox("Customer State", ['SP', 'SC', 'MG', 'RJ', 'RS', 'PA', 'GO', 'ES', 'BA', 'PR', 'MA', 'MS', 'CE', 'RN', 'DF', 'PE', 'MT', 'AM', 'AL', 'RO', 'PB', 'TO', 'PI', 'AC', 'SE', 'RR', 'AP'])
order_status = st.selectbox("Order Status", ['delivered', 'canceled'])
payment_type = st.selectbox("Payment Type", ['credit_card', 'debit_card', 'voucher', 'boleto'])
seller_state = st.selectbox("Seller State", ['SP', 'PR', 'MG', 'ES', 'RS', 'DF', 'SC', 'PE', 'RJ', 'MA', 'BA', 'MT', 'GO', 'MS', 'RO', 'PB', 'CE', 'PA', 'RN', 'PI', 'SE', 'AM'])
product_category = st.selectbox("Product Category", ['Furniture', 'Home & Garden', 'Entertainment', 'Electronics', 'Beauty & Health', 'Books & Stationery', 'Fashion', 'Food & Drinks', 'Industry & Construction'])
text_review = st.selectbox("Is there a Review?", ['yes', 'no'])

# Compile inputs into DataFrame
input_data = pd.DataFrame({
    'customer_state': [customer_state],
    'order_status': [order_status],
    'price': [price],
    'freight_value': [freight_value],
    'product_weight_g': [product_weight_g],
    'payment_type': [payment_type],
    'payment_value': [payment_value],
    'seller_state': [seller_state],
    'product_category': [product_category],
    'product_volume': [product_volume],
    'delivery_duration_days': [delivery_duration_days],
    'estimated_delivery_days': [estimated_delivery_days],
    'ontime_delivery': [ontime_delivery],
    'order_delay': [order_delay],
    'review_length': [review_length],
    'customer_total_orders': [customer_total_orders],
    'text_review': [text_review]
})


input_data_encoded = pd.get_dummies(input_data, drop_first=True)
train_columns = model.feature_names_

missing_cols = set(train_columns) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0

input_data_encoded = input_data_encoded[train_columns]

if st.button("Predict Review"):
    prediction = model.predict(input_data_encoded)
    result = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"Predicted Review Sentiment: {result}")
