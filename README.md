<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=slice&color=191970&fontColor=fff&height=300&section=header&text=SeoulTech%20DataMining%20Project&fontSize=50&rotate=19&fontAlignY=43&fontAlign=57" style="margin-top: -20px;" />
</p>

<div align="center">
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
</div>

# 생활습관 데이터 기반 삶의 질 개선

## Project Info.
> 서울과학기술대학교 데이터마이닝 팀프로젝트

## 분석 배경 및 목적
> + 「경향신문, 지난해 10명 중 8명 만성질환으로 사망···고혈압·당뇨병 유병률 모두 증가」 에 따르면, 주요 만성질환의 유병률은 증가 추세인 것으로 나타난다.
> + 팬데믹 이후로 건강, 삶의 질에 대한 관심도는 계속해서 증가하고 있다.
> + 질병관리청에 따르면, 주관적 건강인지율은 감소하는 추세를 보인다.

## 데이터 수집
> + 「질병관리청, '20 국민건강영양조사」
> + 「질병관리청, '20 원시자료 이용지침서」

## EDA 및 전처리
> 1. 결측치 및 비해당/무응답
> 	+ 공통된 결측치와 모름/무응답 제거
> 	+ 20세 이하 데이터 삭제
> 	+ 대부분의 응답이 비해당인 경우 변수 삭제
> 	+ 결측치가 범주형 변수인 경우 0,1로 대체
> 	+ 결측치가 수치형 변수인 경우 median으로 대체
> 
> 2. (7359, 843)의 데이터셋 중 삶의 질에 큰 영향을 줄 가능성이 있는 변수 20개를 선정  
> + {sex, age, BD1_11, BD2_1, dr_month, BS3_1, E_NWT, BE3_31, BE5_1, pa_aerobic, HE_obe, BM1_0, L_BR_FQ, LS_1YR, BP16_1, BS3_2, BE8_1, BE3_32, HE_BMI}  
>
> 3. 삶의 질을 측정하는데 필요한 변수 선정
> + {D_1_1, LQ_1EQL, LQ_2EQL, LQ_3EQL, LQ_4EQL, LQ_5EQL, EQ5D index, (sub_health, eq5d_100)}
> 
> 4. 삶의 질을 측정할 변수 생성
> + {y_new, y_class}
> <img width="419" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/02f9d10b-d858-4226-b17e-aea8ad9079be">
> <img width="709" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/b09dd9d8-f438-4636-a379-9b3df53ab757">

```python
data = data.drop(data[data['D_1_1'] == 0].index
data['sub_health'] = (100 - 20*(data['D_1_1']-1))

data['eq5d_100'] = 100 * data['EQ5D']

data['y_new'] = ((data['sub_health'] + data['eq5d_100'])/2)/10
data['y_new'] = data['y_new'].round(1)

def classification(data, threshold):
	return np.where(data < threshold, 0, np.where(data == threshold, 1, 2))
	
data['y_class'] = data['y_new'].apply(classification, threshold=8)
```

## 데이터 분석
### SVM / AdaBoost
> + 독립변수<br>
> 'sex', 'age', 'BD1_11', 'BD2_1', 'dr_month', 'BP16_1', 'BS3_1', 'BS3_2', 'BE8_1', 'BE3_31', 'BE3_32', 'BE5_1', 'pa_aerobic', 'HE_BMI', 'HE_obe', 'BM1_0', 'E_NWT', 'L_BR_FQ', 'LS_1YR'
> + 종속변수<br>
> 'y_class'
> <img width="322" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/9b13c2a2-2b7a-483f-a114-6221332e4463">
> 
> + SVM Accuracy : 0.56
> + AdaBoost Accuracy : 0.45

### EQ5D 요인들과 주관적 건강인지
> <img width="632" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/099dec0c-765f-4056-98ce-505db7249106">
> <img width="455" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/6834830b-f908-4edb-aae5-e99d3e28a5c2">



