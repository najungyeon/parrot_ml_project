"""
와인 데이터 전처리 모듈
와인 품질 예측을 위한 데이터 전처리를 수행합니다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

def load_data(train_path='../Data/train.csv', test_path='../Data/test.csv'):
    """데이터 로드"""
    train = pd.read_csv(train_path, encoding='utf-8')
    test = pd.read_csv(test_path, encoding='utf-8')
    return train, test

def handle_missing_values(df):
    """결측값 처리"""
    # 지역 정보 결측값
    for col in ['local1', 'local2', 'local3', 'local4']:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
    
    # 포도 품종 결측값
    for i in range(1, 13):
        col = f'varieties{i}'
        if col in df.columns:
            df[col].fillna('', inplace=True)
    
    # 수치형 결측값
    if 'abv' in df.columns:
        df['abv'].fillna(df['abv'].median(), inplace=True)
    if 'year' in df.columns:
        df['year'].fillna(df['year'].median(), inplace=True)
    if 'price' in df.columns:
        df['price'].fillna(df['price'].median(), inplace=True)
    
    # degree 처리 (온도 범위)
    if 'degree' in df.columns:
        df['degree'].fillna('10~12', inplace=True)
    
    return df

def encode_ordinal_features(df):
    """순서형 범주 변수 인코딩"""
    ordinal_mappings = {
        'sweet': {'SWEET1': 1, 'SWEET2': 2, 'SWEET3': 3, 'SWEET4': 4, 'SWEET5': 5},
        'acidity': {'ACIDITY1': 1, 'ACIDITY2': 2, 'ACIDITY3': 3, 'ACIDITY4': 4, 'ACIDITY5': 5},
        'body': {'BODY1': 1, 'BODY2': 2, 'BODY3': 3, 'BODY4': 4, 'BODY5': 5},
        'tannin': {'TANNIN1': 1, 'TANNIN2': 2, 'TANNIN3': 3, 'TANNIN4': 4, 'TANNIN5': 5}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map(mapping)
            # 결측값 처리 (중간값으로)
            if df[f'{col}_encoded'].isnull().any():
                df[f'{col}_encoded'].fillna(3, inplace=True)
    
    return df

def create_wine_features(df):
    """와인 특성 기반 피처 생성"""
    features_created = []
    
    # 와인 나이 (현재 연도 기준)
    current_year = 2024
    if 'year' in df.columns:
        df['wine_age'] = current_year - df['year']
        features_created.append('wine_age')
    
    # 포도 품종 개수
    variety_cols = [f'varieties{i}' for i in range(1, 13) if f'varieties{i}' in df.columns]
    if variety_cols:
        df['num_varieties'] = df[variety_cols].apply(lambda x: sum(x != ''), axis=1)
        features_created.append('num_varieties')
    
    # 단일 품종 여부
    if 'varieties2' in df.columns:
        df['is_single_variety'] = (df['varieties2'] == '').astype(int)
        features_created.append('is_single_variety')
    
    # 지역 세분화 정도
    local_cols = ['local1', 'local2', 'local3', 'local4']
    if all(col in df.columns for col in local_cols):
        df['location_detail_level'] = df[local_cols].apply(
            lambda x: sum(x != 'Unknown'), axis=1
        )
        features_created.append('location_detail_level')
    
    # 와인 이름에서 특징 추출
    if 'name' in df.columns:
        df['is_reserve'] = df['name'].str.contains('Reserve', case=False, na=False).astype(int)
        df['is_grand_cru'] = df['name'].str.contains('Grand Cru', case=False, na=False).astype(int)
        df['is_special'] = df['name'].str.contains('Special|Limited|Exclusive', case=False, na=False).astype(int)
        features_created.extend(['is_reserve', 'is_grand_cru', 'is_special'])
    
    # 주요 와인 생산국 여부
    if 'nation' in df.columns:
        major_countries = ['France', 'Italy', 'Spain', 'U.S.A', 'Australia', 'Chile', 'Argentina']
        df['is_major_country'] = df['nation'].apply(
            lambda x: any(country in str(x) for country in major_countries)
        ).astype(int)
        features_created.append('is_major_country')
    
    # 가격 범주 (train 데이터만)
    if 'price' in df.columns:
        df['price_category'] = pd.cut(df['price'], 
                                      bins=[0, 20000, 50000, 100000, float('inf')],
                                      labels=['budget', 'mid', 'premium', 'luxury'])
        features_created.append('price_category')
    
    # 온도 범위에서 평균 온도 추출
    if 'degree' in df.columns:
        def extract_avg_temp(degree_str):
            try:
                if pd.isna(degree_str):
                    return 12
                temps = degree_str.split('~')
                if len(temps) == 2:
                    return (float(temps[0]) + float(temps[1])) / 2
                return 12
            except:
                return 12
        
        df['avg_serving_temp'] = df['degree'].apply(extract_avg_temp)
        features_created.append('avg_serving_temp')
    
    print(f"Created {len(features_created)} new features: {features_created}")
    return df

def encode_categorical_features(train_df, test_df):
    """범주형 변수 인코딩"""
    # 인코딩할 범주형 컬럼
    categorical_columns = ['type', 'use', 'nation', 'producer', 
                          'varieties1', 'local1']
    
    le_dict = {}
    
    for col in categorical_columns:
        if col in train_df.columns and col in test_df.columns:
            le = LabelEncoder()
            
            # train과 test의 모든 고유값 합치기
            all_values = pd.concat([train_df[col], test_df[col]]).fillna('Unknown').unique()
            le.fit(all_values)
            
            # 인코딩
            train_df[f'{col}_encoded'] = le.transform(train_df[col].fillna('Unknown'))
            test_df[f'{col}_encoded'] = le.transform(test_df[col].fillna('Unknown'))
            
            le_dict[col] = le
    
    return train_df, test_df, le_dict

def scale_numerical_features(train_df, test_df):
    """수치형 변수 스케일링"""
    numerical_cols = ['price', 'year', 'ml', 'abv', 'wine_age', 'num_varieties', 
                     'location_detail_level', 'avg_serving_temp']
    
    # 실제 존재하는 컬럼만 선택
    numerical_cols = [col for col in numerical_cols if col in train_df.columns]
    
    scaler = StandardScaler()
    
    # train 데이터로 학습
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
    return train_df, test_df, scaler

def preprocess_wine_data(train_path='../Data/train.csv', test_path='../Data/test.csv'):
    """전체 전처리 파이프라인"""
    print("=" * 50)
    print("Starting Wine Data Preprocessing")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("\n1. Loading data...")
    train, test = load_data(train_path, test_path)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # 2. 결측값 처리
    print("\n2. Handling missing values...")
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    # 3. 순서형 변수 인코딩
    print("\n3. Encoding ordinal features...")
    train = encode_ordinal_features(train)
    test = encode_ordinal_features(test)
    
    # 4. 피처 엔지니어링
    print("\n4. Creating wine-specific features...")
    train = create_wine_features(train)
    test = create_wine_features(test)
    
    # 5. 범주형 변수 인코딩
    print("\n5. Encoding categorical features...")
    train, test, le_dict = encode_categorical_features(train, test)
    
    # 6. 수치형 변수 스케일링
    print("\n6. Scaling numerical features...")
    train, test, scaler = scale_numerical_features(train, test)
    
    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    print(f"Final train shape: {train.shape}")
    print(f"Final test shape: {test.shape}")
    
    return train, test, le_dict, scaler

if __name__ == "__main__":
    # 전처리 실행
    train_processed, test_processed, encoders, scaler = preprocess_wine_data()
    
    # 결과 저장
    train_processed.to_csv('../Result/train_wine_processed.csv', index=False, encoding='utf-8')
    test_processed.to_csv('../Result/test_wine_processed.csv', index=False, encoding='utf-8')
    
    print("\nProcessed data saved to Result folder!")
    print("\nSample of processed features:")
    print(train_processed.columns.tolist())