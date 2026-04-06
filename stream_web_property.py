import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. 環境 & モデルをロード ---
st.set_page_config(page_title="Mikawa Property Price AI", layout="wide")

@st.cache_resource
def load_assets():
    # Model, Encoder, KMeans 
    model = joblib.load("mikawa_stacking_model.joblib")
    encoder = joblib.load("mikawa_target_encoder.joblib")
    kmeans = joblib.load("mikawa_kmeans_cluster.joblib")
    return model, encoder, kmeans


model, encoder, kmeans = load_assets()

# --- 2. ユーザーインタフェース ---
st.title("🏠 三河地域不動産価格予測 (AI Model)")
st.write("Stacking Ensemble (XGBoost, LGBM, CatBoost)を使用した不動産価格予測システム")

with st.sidebar:
    st.header("物件概要")
    city = st.selectbox("都市 (City)", ["安城市", "知立市", "蒲郡市", "碧南市", "刈谷市",
                                        "みよし市", "西尾市", "岡崎市", "新城市", "田原市",
                                        "高浜市", "豊橋市", "豊川市", "豊田市"])
    area = st.number_input("敷地面積 (Area m2)", min_value=10.0, max_value=2000.0, value=150.0)
    floor_area = st.number_input("延床面積 (Floor Area m2)", min_value=10.0, max_value=2000.0, value=100.0)
    age = st.slider("築年数 (Building Age)", 0, 60, 15)
    dist = st.number_input("駅からの距離 (分)", 0, 120, 10)
    road = st.number_input("道路の幅 (m)", 2.0, 20.0, 4.0)

# --- 3. 予測実行ボタン ---
if st.button("💰 予測実行"):
    # A. Input data 
    input_data = pd.DataFrame({
        'City': [city],
        'LocationKey': [f"{city}_Unk_Unk"], 
        'Structure': ["木造"], 
        'City_Planning': ["第１種中高層住居専用地域"],
        'Station_Distance': [dist],
        'Area_m2': [area],
        'Floor_Area_m2': [floor_area],
        'Building_Age': [age],
        'Road_Width': [road],
        'Efficiency_ratio': [floor_area / area],
        'Is_New_House': [1 if age <= 5 else 0],
        'Is_Old_House': [1 if age >= 22 else 0],
        'Road_Land_Interaction': [road * area],
        'Log_Area': [np.log1p(area)],
        'Is_Near_Station': [1 if dist <= 10 else 0],
        'Station_Dist_Inv': [1 / (dist + 1)] 
    })

    
    cluster_features = pd.DataFrame({
        'Station_Distance': [dist],
        'Log_Price_per_m2': [12.0] 
    })
    input_data['Area_Cluster'] = kmeans.predict(cluster_features).astype(str)

  
    categorical_cols = ["LocationKey", "City", "Structure", "City_Planning", "Area_Cluster"]
    input_encoded = encoder.transform(input_data[categorical_cols])
    
    
    numeric_cols = ["Station_Distance", "Area_m2", "Floor_Area_m2", "Building_Age", "Road_Width",
                    "Efficiency_ratio", "Is_New_House", "Is_Old_House", "Road_Land_Interaction", 
                    "Log_Area", "Is_Near_Station", "Station_Dist_Inv"]
    
    X_input = pd.concat([input_encoded, input_data[numeric_cols]], axis=1)
    
    
    pred_log_per_m2 = model.predict(X_input)
    price_per_m2 = np.expm1(pred_log_per_m2)[0] 
    
    
    total_price = price_per_m2 * area
    
    
    st.success(f"### 予想取引価格: {total_price:,.0f} 円")
    st.info(f"平米単価 (予想価格/m2): {price_per_m2:,.0f} 円/㎡")