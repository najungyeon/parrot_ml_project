# 전처리(Preprocessing)

ML 모델은 데이터로부터 어떠한 패턴이나 특징을 찾습니다. 
그런데 데이터가 너무 들쭉날쭉이면 패턴을 찾기 힘들지 않을까요?

그래서 우리는 전처리를 해야합니다. 갑자기 너무 큰 수가 들어 있거나 값이 없는 경우가 있다면
이를 제거하거나 보정해주어야 합니다. 

다음 표를 한번 볼까요?
<p align="center">
  <img src="../asset/before_preprocess.jpg" alt="before" width="100%"/>
</p>
앗 Mileage를 이용하여 총 주행 거리를 이용해 중고차의 가격을 추론하려했는데
숫자가 아닙니다! 143131 km와 같이 km이 붙은 문자열이네요.
그럼 이걸 숫자로 바꿔야 합니다. 

다음의 코드를 한번 볼꼐께요. Baseline/data_preprocessing.py 파일에 있는 코드 일부분입니다. 
```python
df["Mileage"] = df["Mileage"].str.replace(" km", "", regex=True).astype(int)
```
이 코드는 df라는 데이터 집합의 Mileage라는 열의 데이터를 가져온 후, 
" km"을 ""(아무것도 없음)으로 대체하는 코드입니다.

# 학습

학습은 데이터로부터 패턴을 찾는 과정입니다. 우리가 준비한 깨끗한 데이터를 모델에게 보여주고,
모델이 스스로 규칙을 찾도록 하는 것이죠.

예를 들어, 중고차 가격 예측 모델은:
- 연식이 오래될수록 가격이 낮아진다
- 주행거리가 많을수록 가격이 낮아진다
- 고급 브랜드일수록 가격이 높다

이런 패턴들을 자동으로 학습합니다.

## 학습 코드 예시
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 데이터 분할 (학습용 80%, 검증용 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_val)
```

# 결과 분석

모델이 얼마나 잘 학습했는지 확인해야 합니다. 
단순히 정확도만 보는 것이 아니라, 어떤 부분에서 실수를 하는지, 
어떤 특징이 가장 중요한지 등을 분석해야 합니다.

## 평가 지표
- **MAE (Mean Absolute Error)**: 예측값과 실제값의 절대 차이의 평균
- **RMSE (Root Mean Squared Error)**: 오차 제곱의 평균의 제곱근
- **R² Score**: 모델이 데이터의 분산을 얼마나 잘 설명하는지

## 시각화를 통한 분석
```python
import matplotlib.pyplot as plt

# 예측값 vs 실제값 플롯
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predictions, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Prediction vs Actual')
plt.show()

# 잔차(residual) 분석
residuals = y_val - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

## Feature Importance 분석
어떤 특징이 가격 예측에 가장 중요한지 확인:
```python
# Random Forest의 경우
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))
```

# 개선 방향

1. **더 나은 전처리**
   - 이상치 제거
   - 스케일링 (MinMaxScaler, StandardScaler)
   - 범주형 변수 인코딩 개선

2. **Feature Engineering**
   - 새로운 특징 생성 (차량 나이, 브랜드별 평균 가격 등)
   - 특징 선택 (불필요한 특징 제거)

3. **고급 모델 사용**
   - Random Forest
   - XGBoost
   - LightGBM

4. **하이퍼파라미터 튜닝**
   - GridSearchCV
   - RandomizedSearchCV
   - Optuna

5. **앙상블 기법**
   - Voting
   - Stacking
   - Blending

이제 여러분만의 방법으로 모델을 개선해보세요! 🚀