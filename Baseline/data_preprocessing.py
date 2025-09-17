"""
데이터 전처리 모듈
중고차 데이터의 전처리를 수행합니다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(train_path, test_path):
    """데이터 로드"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def clean_mileage(df):
    """Mileage 컬럼 정리 - km 단위 제거"""
    if 'Mileage' in df.columns:
        df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False)
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
    return df

def clean_engine_volume(df):
    """Engine volume 정리"""
    if 'Engine volume' in df.columns:
        df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False)
        df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')
    return df

def handle_missing_values(df):
    """결측값 처리"""
    # 수치형 컬럼은 중앙값으로 대체
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 범주형 컬럼은 최빈값으로 대체
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df

def encode_categorical_features(train_df, test_df, categorical_columns):
    """범주형 변수 인코딩"""
    le_dict = {}
    
    for col in categorical_columns:
        if col in train_df.columns:
            le = LabelEncoder()
            # train과 test의 모든 고유값을 합쳐서 학습
            all_values = pd.concat([train_df[col], test_df[col]]).unique()
            le.fit(all_values)
            
            train_df[col + '_encoded'] = le.transform(train_df[col])
            test_df[col + '_encoded'] = le.transform(test_df[col])
            
            le_dict[col] = le
    
    return train_df, test_df, le_dict

def create_features(df):
    """새로운 특징 생성"""
    # 차량 나이 계산 (현재 연도 - 생산 연도)
    current_year = 2024
    if 'Prod. year' in df.columns:
        df['Car_Age'] = current_year - df['Prod. year']
    
    # 주행거리 대비 연식
    if 'Mileage' in df.columns and 'Car_Age' in df.columns:
        df['Mileage_per_Year'] = df['Mileage'] / (df['Car_Age'] + 1)
    
    return df

def preprocess_data(train_path='../Data/train.csv', test_path='../Data/test.csv'):
    """전체 전처리 파이프라인"""
    print("Loading data...")
    train, test = load_data(train_path, test_path)
    
    print("Cleaning data...")
    train = clean_mileage(train)
    train = clean_engine_volume(train)
    test = clean_mileage(test)
    test = clean_engine_volume(test)
    
    print("Handling missing values...")
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    print("Creating features...")
    train = create_features(train)
    test = create_features(test)
    
    # 범주형 컬럼 리스트
    categorical_columns = ['Manufacturer', 'Model', 'Category', 'Leather interior',
                          'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color']
    
    print("Encoding categorical features...")
    train, test, le_dict = encode_categorical_features(train, test, categorical_columns)
    
    print("Preprocessing complete!")
    return train, test, le_dict

if __name__ == "__main__":
    # 전처리 실행
    train_processed, test_processed, encoders = preprocess_data()
    
    # 결과 저장
    train_processed.to_csv('../Result/train_processed.csv', index=False)
    test_processed.to_csv('../Result/test_processed.csv', index=False)
    
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print("\nFirst few rows of processed train data:")
    print(train_processed.head())