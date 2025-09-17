"""
모델 학습 모듈
전처리된 데이터를 사용하여 중고차 가격 예측 모델을 학습합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features(train_df, test_df):
    """모델링을 위한 특징과 타겟 변수 분리"""
    # ID와 Price를 제외한 수치형 컬럼만 선택
    feature_columns = [col for col in train_df.columns 
                      if col not in ['ID', 'Price'] and 
                      ('encoded' in col or train_df[col].dtype in ['int64', 'float64'])]
    
    X = train_df[feature_columns]
    y = train_df['Price']
    X_test = test_df[feature_columns]
    
    return X, y, X_test

def train_linear_regression(X_train, y_train, X_val, y_val):
    """선형 회귀 모델 학습"""
    print("\nTraining Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # 평가
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Validation R2: {val_r2:.4f}")
    
    return model, val_pred

def train_random_forest(X_train, y_train, X_val, y_val):
    """랜덤 포레스트 모델 학습"""
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # 평가
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Validation R2: {val_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    return model, val_pred

def visualize_predictions(y_val, predictions, model_name):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_val, predictions, alpha=0.5)
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_val - predictions
    axes[1].scatter(predictions, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Price')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../Result/{model_name}_predictions.png')
    plt.show()

def create_submission(model, X_test, test_ids, filename):
    """제출 파일 생성"""
    predictions = model.predict(X_test)
    
    # 음수 예측값을 0으로 변경
    predictions = np.maximum(predictions, 0)
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'Price': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"\nSubmission file saved as {filename}")
    return submission

def main():
    """메인 실행 함수"""
    # 전처리된 데이터 로드
    print("Loading preprocessed data...")
    train = pd.read_csv('../Result/train_processed.csv')
    test = pd.read_csv('../Result/test_processed.csv')
    
    # 특징과 타겟 분리
    X, y, X_test = prepare_features(train, test)
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 모델 학습
    lr_model, lr_pred = train_linear_regression(X_train, y_train, X_val, y_val)
    rf_model, rf_pred = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 시각화
    visualize_predictions(y_val, lr_pred, 'Linear_Regression')
    visualize_predictions(y_val, rf_pred, 'Random_Forest')
    
    # 제출 파일 생성
    test_ids = test['ID']
    create_submission(lr_model, X_test, test_ids, '../Result/submission_lr.csv')
    create_submission(rf_model, X_test, test_ids, '../Result/submission_rf.csv')
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()