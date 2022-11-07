# Reinforcement Learning
- RL Lab Intern
- 자율주행 system의 low-level controller를 PID 제어를 통해 설계

## RLLab Intern 22.08 ~
### 2022.08 ~ 2022.09: <ACC, Steering Angle PID 설계>
* metadrive 를 통해 simulation 진행
* low-level에서 목표 acc, steering angle에 대한 pid gain 및 구조들을 우선적으로 설계
* 후에 이들을 활용하여 1) 차선 유지, 2) 차선 멈춤, 3) 차선 변경(왼 or 오) system에 활용할 예정
* 매 순간 어떤 움직임을 가져갈지에 대한 decision making을 ML 차원에서 진행하게 됨.

### 2022.09 ~ : <CPO, safeRL>
* CPO 알고리즘 : constrained Policy Optimization
* continuous한 output을 내던 policy 수정해서 discrete한 output, 즉 decision making만 하도록 수정
* 학습 진행해 본 결과 성능이 굉장히 떨어짐. PID controller를 수정할 필요가 있음.

### 2022.10 ~ : <Behavior cloning>
* raw한 action 학습에 앞서 behavior cloning 후 학습을 진행한 결과 기존의 PID보다 성능이 잘 나옴
* Broken line에 대한 cost가 없어 이를 추가한 뒤 학습을 진행 
  -> 차선을 일정 스텝 이상 동안 넘었을 때만 cost가 있도록 수정
  
* Metadrive code 분석
  -> 각 vehicle 은 각자 navigation object를 가지고 있음(default = node-road 함수 이용) : graph 모델 활용
  -> 시작점부터 이어지는 가장 마지막 블럭(아마 같은 레인과 이어지는 마지막 지점 혹은 노드를 의미하는 걸로 예상됨)을 목적지로 설정 ("auto_assign_task")
  -> 이걸 바탕으로 최단 route(bfs 이용)를 찾아 checkpoints 할당 ("set_route")
  -> 이 체크포인트를 이용하여 navi 의 값들로 활용하는 것으로 보임
  -> Base_navigation의 "_get_info_for_checkpoint"를 확인해보면 설정한 checkpoint 간격을 바탕으로 상대적 거리를 계산하여 정보를 제공함
