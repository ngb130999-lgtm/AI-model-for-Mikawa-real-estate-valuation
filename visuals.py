import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
# Cấu hình Font tiếng Nhật
plt.rcParams['font.family'] = 'IPAGothic' 
plt.rcParams['axes.unicode_minus'] = False

def plot_distributions(y_original, y_log):
    """Vẽ và lưu biểu đồ phân phối trước/sau Log"""
    print("\n📊 価格分布図を作成中...")
    
    # Tạo thư mục images nếu chưa có
    if not os.path.exists('images'): os.makedirs('images')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trước Log
    sns.histplot(y_original, kde=True, color='red', ax=ax1)
    ax1.set_title('元の価格分布 (Before Log)')
    
    # Sau Log
    sns.histplot(y_log, kde=True, color='blue', ax=ax2)
    ax2.set_title('対数変換後の価格分布 (After Log)')
    
    plt.savefig('images/price_distributions.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Vẽ và lưu Top Feature Importance"""
    print("\n🏗️ 特徴量重要度を算出中...")
    
    # Trích xuất tầm quan trọng từ 3 mô hình con
    xgb_imp = model.named_estimators_['xgb'].feature_importances_
    lgb_imp = model.named_estimators_['lgb'].feature_importances_
    cat_imp = model.named_estimators_['cat'].feature_importances_
    
    avg_imp = (xgb_imp/xgb_imp.sum() + lgb_imp/lgb_imp.sum() + cat_imp/cat_imp.sum()) / 3
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_imp
    }).sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('特徴量重要度 (Top 15)')
    
    plt.savefig('images/feature_importance.png', bbox_inches='tight')
    plt.close()