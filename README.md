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
> 2. (7359, 843)의 데이터셋 중 삶의 질에 큰 영향을 줄 가능성이 있는 변수 19개를 선정  
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
### OLS Regression
> + 독립변수<br>
> 'sex', 'age', 'BD1_11', 'BD2_1', 'dr_month', 'BP16_1', 'BS3_1', 'BS3_2', 'BE8_1', 'BE3_31', 'BE3_32', 'BE5_1', 'pa_aerobic', 'HE_BMI', 'HE_obe', 'BM1_0', 'E_NWT', 'L_BR_FQ', 'LS_1YR'
> + 종속변수<br>
> 'y_new'
> <img width="327" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/3c85ea07-2e7d-4986-8842-dd0439a05a12">
> <br>
### SVM / AdaBoost
> + 독립변수<br>
> 'sex', 'age', 'BD1_11', 'BD2_1', 'dr_month', 'BP16_1', 'BS3_1', 'BS3_2', 'BE8_1', 'BE3_31', 'BE3_32', 'BE5_1', 'pa_aerobic', 'HE_BMI', 'HE_obe', 'BM1_0', 'E_NWT', 'L_BR_FQ', 'LS_1YR'
> + 종속변수<br>
> 'y_class'
> <img width="322" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/9b13c2a2-2b7a-483f-a114-6221332e4463">
> 
> + SVM Accuracy : 0.56
> + AdaBoost Accuracy : 0.45
> <br>

### EQ5D 요인들과 주관적 건강인지
> <img width="632" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/099dec0c-765f-4056-98ce-505db7249106">
> <img width="455" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/6834830b-f908-4edb-aae5-e99d3e28a5c2">
> + Feature Importance가 높은 LQ_3EQL, LQ_1EQL를 분석(LQ_4EQL는 분석할 세부 Feature 부족)
> <br>

### GradientBoosting - LQ_3EQL(일상생활)
> 1. 변수 설정 및 Encoding, Scaling
> >	+ 독립변수 - 일상생활 관련 변수
> >	1. BP16_1: 주중 하루 평균 수면시간
> >	2. E_NWT: 하루평균 근거리 작업시간
> >	3. L_BR_FQ: 최근 1년 동안 1주동안 아침식사 빈도
> >	4. LS_1YR: 최근 1년 동안 2주 이상 식이보충제 복용 여부
> >	5. BD1_11: 1년간 음주빈도
> >	6. BS3_1: 현재 일반담배 흡연 여부
> <img width="520" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/3355a29b-365b-48f4-a3ff-05023dd32c0f">
> 2. 모델링
> <img width="868" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/1a803aad-eace-4d87-838a-cc6f8d34dbb3">
> <img width="854" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/61216f1d-4fc1-4f85-9936-ebbc0c220e2d">
> 3. 분석결과
> + LQ_3EQL(일상활동 관련 삶의 질)에 가장 많이 영향을 미치는 요인 파악 
> <img width="787" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/7851a516-6d38-4163-90e0-c2b3c7f7fdbc">
> + E_NWT(근거리 작업 시간)과 BP16_1(주중 하루 평균 수면시간)이 가장 많은 영향을 준다.
> <br>

### GradientBoosting - LQ_1EQL(운동능력)
> 1. 변수 설정 및 Encoding, Scaling
> >	+ 독립변수 - 운동능력 관련 변수
> >	1. BE3_31: 1주일 간 걷기 일수
> >	2. BE3_ 32: 걷기 지속시간
> >	3. BE8_1: 평소 하루 앉아서 보내는 시간
> >	4. BE5_1: 1주일 간 근력운동 일수
> >	5. pa_aerobic: 유산소 신체활동 실천율
> >	6. HE_obe: 비만 유병여부
> <img width="524" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/9741be39-b9ec-4772-b1e7-a9e0dfeff093">
> 2. 모델링
> <img width="857" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/8e958d32-7e0c-4b31-b07a-2a3d14383d0d">
<img width="800" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/746a8505-0fef-4c2e-aa39-11ed6ad80bc3">
> 3. 분석결과
> + LQ_1EQL(신체활동 관련 삶의 질)에 가장 많이 영향을 미치는 요인 파악
> <img width="796" alt="image" src="https://github.com/JinjangKang/datamining_repo/assets/117068666/82343d1c-7953-4d28-8321-b3f7b6543ce9">
> + BE8_1(평소 하루 앉아서 보내는 시간)과 BE3_31(1주일 간 걷기 일수)이 가장 많은 영향을 준다.

## 분석결과
#### 앉아있는 시간이 건강에 영향을 크게 미친다.
#### 흡연과 음주가 생각보다 매우 큰 영향력을 가지지 않으며, 근력운동보다 걷기운동이 건강에 더 큰 영향을 미친다.

## 한계점 및 추후 개선방안
> + 중요 요소의 자세한 값 파악 불가(ex) 수면시간, 앉아있는 시간)<br>
> + LQ-4EQL 요소 파악 불가<br>
> + EQ-5D 지표 사용 불가<br>
> + 데이터에 미응답과 결측치가 많음
> + 향후 2019, 2021년도 데이터를 추가 분석한다면 더 면밀한 분석 가능 예상

## Reference
> + KOSIS 국가통계포털(질병관리청, 국가건강영양조사, 주관적 건강인지률 추이)
> + image designed by Freepik  
> + KOSIS 국가통계포털(문화체육관광부, 국민생활체육조사, 건강 및 체력 유지 방법 수행정도(충분한 휴식 및 수면))
> + 민서영 기자, 지난해 10명 중 8명 만성질환으로 사망···고혈압·당뇨병 유병률 모두 증가, 경향신문, 2022.10.17
> + 이영재 기자, 젊은층 당뇨병 유병률 증가…선별검사 나이 40→35세로 낮춰야 경향신문, 2023.01.19




