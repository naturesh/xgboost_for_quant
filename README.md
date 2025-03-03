# xgboost_for_quant
하이킨아시 시그널 분류를 통한 매수 포인트 예측 모델
<br><br><br>
## 하이킨 아시의 단점을 보완하자

하이킨아시는 추세매매에서는 강세를 보이는 지표라고 할수 있습니다. <br>
하지만 이를 횡보장에서 이용하게 될경우 매우 큰 손실을 안겨줄수 있습니다.
따라서 앞으로 n%이상 상승이 확정된 하이킨아시 시그널을 얻고자 모델을 제작했습니다.
<br><br><br>

---

## XGBOOST 분류 모델을 통한 하이킨아시 추세 시그널 체크 

- 하이킨아시의 시그널이 나온 이후 미래 100분봉 데이터 평균값이 현재 값보다 n% 이상 상승한 경우

- 하이킨아시의 시그널이 나온 이후 미래 100분봉 데이터 평균값이 현재 값보다 n% 이상 하락한 경우

- 하이킨아시의 시그널이 나온 이후 미래 100분봉 데이터 평균값이 현재 값보다 k1% 이상 상승 k2% 이상 하락한 경우

<br>

3개의 레이블로 나누어 학습을 진행했습니다. 
<br><br><br>
# Train Dataset

RSI, EMA, SMA, CCI 등의 지표와 하이킨아시 캔들차트에서의 RSI, EMA, SMA, CCI 지표를 섞어 사용했습니다.

약 6개의 코인, 3년간의 데이터를 통해 학습을 진행한 결과 71%의 정확도를 보여주는 모델을 제작하게 되었습니다.
<br><br><br>

![image](https://github.com/user-attachments/assets/5cde65de-a0da-4bc7-9300-061c5b909e32)


### 매수포인트 예측모델인데 왜 삼진분류 멀티클래스 모델일까?

- 이에 관해서는 저도 의문입니다.
- 매수 포인트 예측과 트레일링 스탑 기법을 활용해 퀀트 투자를 구상하였습니다
- 매수, Or not 매수 포인트로 예측을 진행한 결과 현재 71% 보다 훨씬 낮은 예측률을 기록할수밖에 없었습니다.
- 따라서 현재 3진분류 모델로 이용하고 있습니다.


<br><br><br>

# 결과 공유 2025-01-01 ~ 2025-02-28 (모델 예측 결과 + 하락 추세 조건문)

TRX
![trx](https://github.com/user-attachments/assets/4eebf7ff-9ddf-4cec-8f63-f4820ae16e47)
<br><br><br>

BTC
![btc](https://github.com/user-attachments/assets/f951efa6-366f-4bfd-93d1-03ead1700b17)
<br><br><br>

SOL
![sol](https://github.com/user-attachments/assets/66bb1d1f-22c2-4134-a9ae-2f2915062efc)
<br><br><br>

ETC
![newplot](https://github.com/user-attachments/assets/0a04a2ff-0c39-4152-8d9b-f96c04b290ab)
<br><br><br>


# 사용 방법

train.ipynb 마지막 ploty 차트 부분 참고 (train.ipynb의 finder 함수 이용) (ipynb 파일을 py로 export 후 finder 이용 가능)

```python

model = xgb.XGBClassifier()
model.load_model('path/model.json')
```

