# 🍷 와인의 품격을 예측하다

<p align="left">
  <img src="../asset/wine.jpg" width="300"/>
</p>

### 🍇 소믈리에: _"좋은 와인은 숫자로 말한다"_  

전 세계 와인의 특성을 분석하여,  
**와인 품질 예측 모델을 만들어 주세요!** 🍾  

---

## 🎯 **프로젝트 개요**  
🔍 **Mission**  
- 주어진 `train.csv` 데이터를 이용해 **와인 품질 예측 모델**을 만드세요.  
- `test.csv` 데이터를 기반으로 예측값을 도출하고 `submission.csv`에 작성하세요.  

📊 **Dataset 구조**  
- **train 데이터**: `7201 rows × 31 columns`  
- **test 데이터**: `1800 rows × 31 columns`  
- **submission.csv**: `1800 rows × 2 columns` (목표 변수 `quality`는 0으로 채워져 있음)  

---

## 🍇 **데이터 설명 (Feature Dictionary)**  

### 주요 특성
| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| **id** | 와인 고유번호 | 162064 |
| **quality** _(목표 변수)_ | 와인 품질 점수 | 예측 대상 |
| **name** | 와인 이름 | "Wrongo Dongo" |
| **producer** | 생산자/와이너리 | "보데가스 볼베르" |
| **nation** | 생산 국가 | "스페인 Spain" |
| **type** | 와인 종류 | Red, White, Sparkling |
| **price** | 가격 (KRW) | 34000 |
| **year** | 빈티지 (생산연도) | 2016 |

### 맛 특성
| 컬럼명 | 설명 | 값 범위 |
|--------|------|---------|
| **sweet** | 단맛 정도 | SWEET1(드라이) ~ SWEET5(스위트) |
| **acidity** | 산도 | ACIDITY1 ~ ACIDITY5 |
| **body** | 바디감 | BODY1(라이트) ~ BODY5(풀바디) |
| **tannin** | 탄닌 | TANNIN1 ~ TANNIN5 |

### 기타 정보
| 컬럼명 | 설명 |
|--------|------|
| **varieties1~12** | 포도 품종 (최대 12개) |
| **local1~4** | 생산 지역 (대→소 분류) |
| **use** | 권장 용도 (Table, Appetizer 등) |
| **abv** | 알코올 도수 (%) |
| **degree** | 음용 온도 (°C) |
| **ml** | 용량 (밀리리터) |

---

## 🎯 **평가 지표**  
**MAE (Mean Absolute Error)**  
```python
MAE = (1/n) * Σ|실제값 - 예측값|
```
낮을수록 좋은 점수입니다!

---

## 💡 **도전 과제**  

### 🥉 **Bronze Level**
- 데이터 전처리 완료
- 기본 모델(Linear Regression) 구현
- 정확도 60% 이상

### 🥈 **Silver Level**  
- Feature Engineering 적용
- 고급 모델(Random Forest, XGBoost 등) 사용
- 정확도 70% 이상

### 🥇 **Gold Level**
- 앙상블 기법 활용
- 교차 검증 구현
- 정확도 80% 이상

---

## 📝 **힌트**  

1. **결측값 처리**: `local1~4`, `varieties2~12` 등에 결측값이 많아요
2. **범주형 변수**: 순서형 변수(sweet, acidity 등)는 순서를 유지하며 인코딩
3. **Feature Engineering**: 
   - 와인 나이 (현재 - 빈티지)
   - 포도 품종 개수
   - 단일 품종 여부
   - 주요 생산국 여부
4. **텍스트 활용**: 와인 이름에서 'Reserve', 'Grand Cru' 등 키워드 추출

---

## 🚀 **시작하기**  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (데이터에 한글 포함)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
train = pd.read_csv('Data/train.csv', encoding='utf-8')
test = pd.read_csv('Data/test.csv', encoding='utf-8')
submission = pd.read_csv('Data/submission.csv')

# 데이터 확인
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\n컬럼 리스트:")
print(train.columns.tolist())
print("\n첫 5개 샘플:")
print(train.head())

# 와인 종류별 분포
print("\n와인 종류별 개수:")
print(train['type'].value_counts())
```

---

**화이팅! 여러분의 멋진 모델을 기대합니다!** 🎉