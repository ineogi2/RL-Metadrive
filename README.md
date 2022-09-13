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

### Git  
* git init
* ssh-keygen - public / private key 생성
* 해당 public key를 github ssh key 등록 칸에 등록해준다
* git clone <ssh용>

* git add . - 현재 폴더 전부 등록  
* git status - 상태 확인  
* git commit -m "commit message" - commit 등록  
* git push origin main - 업로드  

### Ubuntu  
#### -복사
* cp [option] <원본> <사본>  
  * -p : 원본 그대로 복사  
  * -r : 하위 디렉토리 및 파일 복사  
  
#### -이동
* mv [option] <원본> [디렉토리]  

#### -폴더 생성
* mkdir -p [디렉토리]  

#### -삭제
* rm [option] <파일 or 폴더>  
  * -f : 바로 삭제
  * -r : 디렉토리도 삭제 가능
  * -rf : 디렉토리까지 포함하여 전부 삭제
* sudo apt-get purge <package_name> : 특정 패키지 삭제

#### -부팅
* sudo reboot : 재부팅
* sudo shutdown -h now : 바로 종료
