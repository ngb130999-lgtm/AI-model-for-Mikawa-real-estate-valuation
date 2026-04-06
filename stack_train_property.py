import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from visuals import plot_distributions, plot_feature_importance

# --- 1. 設定 & データ読み込み ---
FILE_X_TRAIN = r"/home/giabao_1309/property/mikawa_X_train_final.csv"
FILE_Y_TRAIN = r"/home/giabao_1309/property/mikawa_y_train.csv"
FILE_X_TEST = r"/home/giabao_1309/property/mikawa_X_test_final.csv"
FILE_Y_TEST = r"/home/giabao_1309/property/mikawa_y_test.csv"



def train_and_save_model():
    print("🚀 データを読み込み中...")
    X_train = pd.read_csv(FILE_X_TRAIN)
    y_train = pd.read_csv(FILE_Y_TRAIN).values.flatten()
    X_test = pd.read_csv(FILE_X_TEST)
    y_test = pd.read_csv(FILE_Y_TEST).values.flatten()
    
    plot_distributions(np.expm1(y_test), y_test)



    # --- 2. フェーズ 1: ノイズ除去 (Residual Filtering) ---
    print("\n🧹 データのノイズ除去を実行中 (フェーズ 1)...")
    # 残差を算出するための一時的なモデルを学習
    model_temp = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, tree_method='hist')
    model_temp.fit(X_train, y_train)
    
    y_train_pred_log = model_temp.predict(X_train)
    residuals = np.abs(y_train - y_train_pred_log)
    
    # 誤差の大きい上位5%を外れ値として除外
    threshold = np.percentile(residuals, 95)
    mask = residuals < threshold
    X_train_cleaned = X_train[mask]
    y_train_cleaned = y_train[mask]
    print(f"✅ {sum(~mask)} 件のノイズデータを除外しました。残りのデータ数: {X_train_cleaned.shape[0]} 件")

    # --- 3. ベースモデルの定義 (Base Models) ---
    print("\n🏗️ スタッキングシステムを構築中 (XGBoost + LightGBM + CatBoost)...")
    
    # モデル 1: XGBoost
    model_xgb = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=2000, # スタッキング時の過学習を防ぐため調整
        learning_rate=0.02,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        tree_method='hist'
    )

    # モデル 2: LightGBM
    model_lgb = lgb.LGBMRegressor(
        objective='regression_l1', # MAEを最適化
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        random_state=42,
        verbose=-1
    )

    # モデル 3: CatBoost
    model_cat = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        random_state=42,
        verbose=0
    )

    # --- 4. スタッキングモデルの学習 (Final Model) ---
    # メタモデル: Ridge回帰（正則化線形回帰）を使用して各モデルの予測値を統合
    estimators = [
        ('xgb', model_xgb),
        ('lgb', model_lgb),
        ('cat', model_cat)
    ]
    
    final_stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, # 内部的なクロスバリデーションでメタモデルの過学習を防止
        n_jobs=-1 # 並列処理で計算時間を短縮
    )

    print("⏳ アンサンブルモデルを学習中 (数分かかる場合があります)...")
    final_stacking_model.fit(X_train_cleaned, y_train_cleaned)

    # --- 5. 最終評価 ---
    y_pred_log = final_stacking_model.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print("\n" + "="*40)
    print(f"📊 スタッキングモデル 最終評価結果 (ノイズ除去後):")
    print(f"📍 平均絶対誤差 (MAE): {mae_actual:,.0f} 円")
    print(f"📍 決定係数 (R2 Score): {r2:.4f}")
    print("="*40)
    
    plot_feature_importance(final_stacking_model, X_train.columns)

    # --- 6. モデルの保存 ---
    model_filename = "mikawa_stacking_model.joblib"
    joblib.dump(final_stacking_model, model_filename)
    print(f"\n💾 スタッキングモデルを保存しました: {model_filename}")

if __name__ == "__main__":
    train_and_save_model()