"""
탐색적 데이터 분석 (EDA) 모듈
데이터의 분포와 특성을 시각화하고 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_describe_data(train_path='../Data/train.csv', test_path='../Data/test.csv'):
    """데이터 로드 및 기본 정보 출력"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("\nColumn types:")
    print(train.dtypes.value_counts())
    
    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    missing_train = train.isnull().sum()
    missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
    print("Train dataset:")
    for col, count in missing_train.items():
        print(f"  {col}: {count} ({count/len(train)*100:.2f}%)")
    
    missing_test = test.isnull().sum()
    missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
    print("\nTest dataset:")
    for col, count in missing_test.items():
        print(f"  {col}: {count} ({count/len(test)*100:.2f}%)")
    
    return train, test

def analyze_target_variable(train):
    """타겟 변수 (Price) 분석"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 히스토그램
    axes[0].hist(train['Price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 로그 변환 히스토그램
    price_log = np.log1p(train['Price'])
    axes[1].hist(price_log, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('log(Price + 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-transformed Price Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # 박스플롯
    axes[2].boxplot(train['Price'])
    axes[2].set_ylabel('Price')
    axes[2].set_title('Price Boxplot')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../Result/price_distribution.png')
    plt.show()
    
    print("\n" + "=" * 50)
    print("PRICE STATISTICS")
    print("=" * 50)
    print(train['Price'].describe())
    print(f"\nSkewness: {train['Price'].skew():.2f}")
    print(f"Kurtosis: {train['Price'].kurtosis():.2f}")

def analyze_numerical_features(train):
    """수치형 변수 분석"""
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['ID', 'Price']]
    
    # 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    corr_matrix = train[numerical_cols + ['Price']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../Result/correlation_heatmap.png')
    plt.show()
    
    # Price와의 상관관계 Top 10
    price_corr = corr_matrix['Price'].sort_values(ascending=False)
    print("\n" + "=" * 50)
    print("TOP 10 FEATURES CORRELATED WITH PRICE")
    print("=" * 50)
    print(price_corr.head(11))  # Price 자체 포함
    
    # 주요 수치형 변수 분포
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    important_features = ['Prod. year', 'Engine volume', 'Mileage', 
                         'Cylinders', 'Airbags', 'Levy']
    
    for idx, col in enumerate(important_features):
        if col in train.columns and idx < len(axes):
            axes[idx].hist(train[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../Result/numerical_features_distribution.png')
    plt.show()

def analyze_categorical_features(train):
    """범주형 변수 분석"""
    categorical_cols = train.select_dtypes(include=['object']).columns
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for idx, col in enumerate(categorical_cols[:9]):
        # Top 10 categories
        top_categories = train[col].value_counts().head(10)
        axes[idx].barh(range(len(top_categories)), top_categories.values)
        axes[idx].set_yticks(range(len(top_categories)))
        axes[idx].set_yticklabels(top_categories.index)
        axes[idx].set_xlabel('Count')
        axes[idx].set_title(f'Top 10 {col}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../Result/categorical_features_distribution.png')
    plt.show()
    
    # 카테고리별 평균 가격
    print("\n" + "=" * 50)
    print("AVERAGE PRICE BY CATEGORY")
    print("=" * 50)
    
    for col in ['Manufacturer', 'Category', 'Fuel type', 'Gear box type']:
        if col in train.columns:
            avg_price = train.groupby(col)['Price'].mean().sort_values(ascending=False)
            print(f"\n{col} (Top 5):")
            for cat, price in avg_price.head().items():
                print(f"  {cat}: ${price:,.0f}")

def analyze_relationships(train):
    """변수 간 관계 분석"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price vs Production Year
    axes[0, 0].scatter(train['Prod. year'], train['Price'], alpha=0.5)
    axes[0, 0].set_xlabel('Production Year')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].set_title('Price vs Production Year')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price vs Mileage
    mileage_clean = pd.to_numeric(train['Mileage'].str.replace(' km', ''), errors='coerce')
    axes[0, 1].scatter(mileage_clean, train['Price'], alpha=0.5)
    axes[0, 1].set_xlabel('Mileage (km)')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].set_title('Price vs Mileage')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Price by Category
    train.boxplot(column='Price', by='Category', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].set_title('Price Distribution by Category')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)
    
    # Price by Fuel Type
    train.boxplot(column='Price', by='Fuel type', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Fuel Type')
    axes[1, 1].set_ylabel('Price')
    axes[1, 1].set_title('Price Distribution by Fuel Type')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../Result/price_relationships.png')
    plt.show()

def identify_outliers(train):
    """이상치 탐지"""
    print("\n" + "=" * 50)
    print("OUTLIER DETECTION")
    print("=" * 50)
    
    # Price outliers (IQR method)
    Q1 = train['Price'].quantile(0.25)
    Q3 = train['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    price_outliers = train[(train['Price'] < lower_bound) | (train['Price'] > upper_bound)]
    print(f"Price outliers: {len(price_outliers)} ({len(price_outliers)/len(train)*100:.2f}%)")
    print(f"  Lower bound: ${lower_bound:,.0f}")
    print(f"  Upper bound: ${upper_bound:,.0f}")
    
    # 극단적인 가격 예시
    print("\nExtreme prices:")
    print("Top 5 most expensive:")
    print(train.nlargest(5, 'Price')[['ID', 'Manufacturer', 'Model', 'Prod. year', 'Price']])
    print("\nTop 5 cheapest (non-zero):")
    print(train[train['Price'] > 0].nsmallest(5, 'Price')[['ID', 'Manufacturer', 'Model', 'Prod. year', 'Price']])

def main():
    """메인 실행 함수"""
    # 데이터 로드 및 기본 정보
    train, test = load_and_describe_data()
    
    # 타겟 변수 분석
    analyze_target_variable(train)
    
    # 수치형 변수 분석
    analyze_numerical_features(train)
    
    # 범주형 변수 분석
    analyze_categorical_features(train)
    
    # 변수 간 관계 분석
    analyze_relationships(train)
    
    # 이상치 탐지
    identify_outliers(train)
    
    print("\n" + "=" * 50)
    print("EDA COMPLETE!")
    print("=" * 50)
    print("Check the Result folder for saved visualizations.")

if __name__ == "__main__":
    main()