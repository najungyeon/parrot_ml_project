# 📊 와인 데이터셋 설명서

## 🍷 데이터셋 개요

이 데이터셋은 전 세계 와인의 다양한 특성과 가격 정보를 담고 있습니다. 
여러분의 목표는 와인의 특성을 바탕으로 **와인 품질(quality)**을 예측하는 모델을 만드는 것입니다!

### 📁 데이터 구성
- **train.csv**: 학습용 데이터 (7,201개 와인)
- **test.csv**: 테스트용 데이터 (1,800개 와인)
- **submission.csv**: 제출 파일 템플릿
- **leaderboard.csv**: 리더보드용 데이터

---

## 🏷️ 컬럼 설명 (Feature Dictionary)

### 🆔 식별 정보
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| **id** | 와인의 고유 식별번호 | Integer |
| **name** | 와인 이름 | String |
| **producer** | 생산자/와이너리 이름 | String |

### 🌍 지역 정보
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| **nation** | 생산 국가 | String |
| **local1** | 생산 지역 (대분류) | String |
| **local2** | 생산 지역 (중분류) | String |
| **local3** | 생산 지역 (소분류) | String |
| **local4** | 생산 지역 (세부) | String |

### 🍇 포도 품종
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| **varieties1~12** | 포도 품종 1~12 | String |
| | 와인에 사용된 포도 품종들 (최대 12개) | |
| | 앞 번호일수록 비중이 높음 | |

### 🍾 와인 특성
| 컬럼명 | 설명 | 데이터 타입 | 값 범위 |
|--------|------|------------|---------|
| **type** | 와인 종류 | String | Red, White, Sparkling, Rose, Fortified, Dessert |
| **use** | 권장 용도 | String | Table, Appetizer, Dessert, Digestif |
| **abv** | 알코올 도수 | Float | 일반적으로 9~16% |
| **degree** | 음용 온도 | String | 예: "16~18" (°C) |

### 👅 맛 특성
| 컬럼명 | 설명 | 데이터 타입 | 값 범위 |
|--------|------|------------|---------|
| **sweet** | 단맛 정도 | String | SWEET1(드라이) ~ SWEET5(스위트) |
| **acidity** | 산도 | String | ACIDITY1(낮음) ~ ACIDITY5(높음) |
| **body** | 바디감 | String | BODY1(라이트) ~ BODY5(풀바디) |
| **tannin** | 탄닌 | String | TANNIN1(낮음) ~ TANNIN5(높음) |

### 💰 기타 정보
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| **price** | 가격 (원화) | Float |
| **year** | 빈티지 (생산연도) | Float |
| **ml** | 용량 (밀리리터) | Float |

### 🎯 타겟 변수
| 컬럼명 | 설명 | 데이터 타입 | 값 범위 |
|--------|------|------------|---------|
| **quality** | 와인 품질 점수 | Integer | 예측 대상 (submission에서) |

---

## 📈 데이터 특징

### 결측값
- 지역 정보 (local1~4)에 결측값이 많음
- 포도 품종 (varieties2~12)도 일부만 채워져 있음
- 일부 와인의 abv, degree 정보가 누락됨

### 범주형 변수
- **nation, local1~4**: 지역 정보
- **varieties1~12**: 포도 품종
- **type, use**: 와인 분류
- **sweet, acidity, body, tannin**: 순서형 범주 변수

### 수치형 변수
- **abv**: 알코올 도수
- **price**: 가격
- **year**: 빈티지
- **ml**: 용량

### 텍스트 변수
- **name**: 와인 이름
- **producer**: 생산자
- **degree**: 온도 범위 (문자열로 저장됨)

---

## 💡 데이터 전처리 힌트

### 1. 결측값 처리
```python
# 지역 정보 결측값을 'Unknown'으로 대체
df['local1'].fillna('Unknown', inplace=True)

# 포도 품종 결측값은 빈 문자열로
for i in range(2, 13):
    df[f'varieties{i}'].fillna('', inplace=True)
```

### 2. 범주형 변수 인코딩
```python
# 순서형 변수는 순서를 유지하며 인코딩
sweet_map = {'SWEET1': 1, 'SWEET2': 2, 'SWEET3': 3, 'SWEET4': 4, 'SWEET5': 5}
df['sweet_encoded'] = df['sweet'].map(sweet_map)
```

### 3. Feature Engineering 아이디어
- **포도 품종 수**: 사용된 포도 품종의 개수
- **와인 나이**: 현재 연도 - 빈티지
- **지역 세분화 정도**: local 정보가 얼마나 세분화되어 있는지
- **단일 품종 여부**: varieties2가 비어있으면 단일 품종
- **프리미엄 여부**: 가격 기준으로 구분
- **주요 생산국**: 프랑스, 이탈리아, 스페인 등 주요 생산국 여부

### 4. 텍스트 데이터 활용
```python
# 와인 이름에서 특정 키워드 추출
df['is_reserve'] = df['name'].str.contains('Reserve', case=False, na=False)
df['is_grand_cru'] = df['name'].str.contains('Grand Cru', case=False, na=False)
```

---

## 🎯 평가 지표

모델의 성능은 다음 지표로 평가됩니다:
- **MAE (Mean Absolute Error)**: 예측값과 실제값의 절대 오차 평균
- **RMSE (Root Mean Squared Error)**: 큰 오차에 더 큰 페널티
- **R² Score**: 모델의 설명력

---

## 🚀 시작하기

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

# 기본 정보 확인
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\n컬럼 정보:")
print(train.info())

# 타겟 변수 분포 확인 (train에 quality 컬럼이 있다면)
# plt.figure(figsize=(10, 6))
# train['quality'].value_counts().sort_index().plot(kind='bar')
# plt.title('와인 품질 분포')
# plt.xlabel('Quality')
# plt.ylabel('Count')
# plt.show()
```

---

## 📝 추가 참고사항

1. **한글 인코딩**: 데이터에 한글이 포함되어 있으므로 인코딩에 주의
2. **가격 단위**: price는 원화(KRW) 단위
3. **온도 정보**: degree는 문자열 형태 (예: "16~18")로 파싱 필요
4. **블렌딩 와인**: 여러 포도 품종이 섞인 경우 varieties1부터 순서대로 비중이 높음

---

**Good Luck with Your Wine Quality Prediction! 🍷✨**