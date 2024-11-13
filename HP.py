import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载模型
model = joblib.load('CatBoost.pkl')

# Streamlit的用户界面
st.title("Sodium borohydride hydrogen production prediction platform")

# age: 数值输入
LoadingRate = st.number_input("Catalyst Loading Rate(%)", min_value=58, max_value=100, value=70)

# trestbps: 数值输入
ReactionTime = st.number_input("Reaction Time(min):", min_value=1, max_value=51, value=20)

# chol: 数值输入
Temperature = st.number_input("Temperature(K):", min_value=303, max_value=363, value=320)

# thalach: 数值输入
CatalystDosage = st.number_input("Catalyst Dosage(g):", min_value=0.01, max_value=0.05, value=0.03)

# oldpeak: 数值输入
NaBH4 = st.number_input("NaBH4(mol/L):", min_value=0.5, max_value=5.0, value=3.0)

# ca: 数值输入
NaOH = st.number_input("NaOH(mol/L):", min_value=0.25, max_value=4.00, value=2.00)

# 处理输入并进行预测
feature_values = [LoadingRate, ReactionTime, Temperature, CatalystDosage, NaBH4, NaOH]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_proba = model.predict(features)

    # 显示预测结果
    st.write(f"**Predicted hydrogen production:** {predicted_proba}")
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
