#numpy, matplotlib.pyplot 라이브러리를 np와 plt로 입력
import numpy as np 
import matplotlib.pyplot as plt

#실제 라인 값 x y
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#w=1 #무작위 값 w

#linear한 함수 정의 -> forward(3)이면 3w의 값이 나옴
def forward(x):
  return x*w 

#loss 함수 정의 { 예측한 y와 실제 y의 거리차의 제곱 } -> loss(2,3)이면 (2w-3)^2의 값이 나옴
def loss(x,y):
  y_prediction = forward(x)
  return (y_prediction-y)**2  

#w와 MSE값을 넣을 리스트 생성
w_list = []
MSE_list = []  

for w in np.arange(0.0, 4.5, 0.5): #w의 값을 0에서부터 4.0까지 0.5의 간격으로 올림
  print("w=", w) #w가 무슨 값인지 보여주고 시작
  l_sum = 0
  for x_val, y_val in zip(x_data, y_data): #x값 y값을 데이터에서 병렬로 꺼내옴(첫번째, 두번째 세로줄)
    y_prediction_val = forward(x_val) #w에 따라 바뀌는 예측값인 y^을 정의(세번째 세로줄)
    l = loss(x_val, y_val) #l을 (y^-y)의 제곱 즉, 두번째 줄과 세번째 줄의 차의 제곱으로 설정(네번째 세로줄)
    l_sum += l #loss 값들을 모두 합침
    print("\t", x_val, y_val, y_prediction_val, l)

  print("MSE=", l_sum/len(x_data)) #loss값들의 평균을 출력
  w_list.append(w) #w리스트에 계속 w값을 추가
  MSE_list.append(l_sum/len(x_data)) #MSE리스트에 계속 MSE값을 추가

plt.plot(w_list, MSE_list) #두 리스트를 각각 x, y축으로 설정하는 함수 생성
plt.ylabel('Loss') #y축을 Loss라고 지정
plt.xlabel('w') #x축을 w라고 지정
plt.show() #그래프 보여주기
