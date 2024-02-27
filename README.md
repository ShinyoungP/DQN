## **DQN**

### **DQN과 Q-learning의 차이**
- **Q-learning** : 벨만 방정식을 사용해 반복적인 업데이트를 통해 Q-값을 예측한다.
- **DQN** : Q-러닝의 개념을 심층 신경망에 적용한 것이다. 이 방법은 입력으로 입력 상태를 받아 각 행동에 대한 Q-값을 출력으로 반환하는 ANN(인공신경망)을 사용해 Q-값을 예측한다.
  
<br>

### **DQN 알고리즘의 주요 특징**
**1. 경험 리플레이** : 에이전트가 환경에서 탐험하면서 얻는 샘플(s,a,r,s')들을 메모리에 저장했다가 에이전트가 학습할 때 모인 샘플들 중 여러 개의 샘플을 무작위로 뽑아 인공신경망을 업데이트한다. 리플레이 메모리는 크기가 정해져 있어서 메모리가 꽉 차면 가장 먼저 들어온 샘플부터 메모리에서 삭제한다.

<br>

![](https://tave-6th-rlstudy.github.io/assets/img/RL_Study6/dqn.png)

<br>

**2. 타겟 네트워크** 
- **타겟 값** : 실제 값이거나 이상적으로 원하는 예측
$$R(s_t,a_t)+\gamma \max_{a} (Q(s_{t+1}, a))$$
- **주 네트워크** : 현재 상태에서의 Q-값을 예측
- **타겟 네트워크** : 타겟 값을 계산
- **손실 함수** : 학습 과정에서 신경망의 예측이 실제 값(타겟 값)과 얼마나 차이나는지를 측정하는 데 사용된다. 이 함수를 이용해 예측과 타겟의 오차를 줄여 나간다.
$$\mathbb{E}[(타겟-Q(s_t,a_t))^2]$$

<br>

![](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcw2t8w%2FbtrDU3Srd1K%2FDs7UlI3vY9qEhYqNtXSqi0%2Fimg.jpg)

### **전체 알고리즘**
1. 현재 상태 $s_t$의 Q-값을 예측한다.
2. 선택된 행동 $a_t$를 수행한다.
3. 보상 $R(s_t,a_t)$을 받는다.
4. 다음 상태 $s_{t+1}$에 도달한다.
5. 메모리에 전이 $(s_t,a_t,r_t,s_{t+1})$을 추가한다.
6. 메모리에서 무작위로 선택한 전이로 배치 B를 구성한다. 무작위 배치 B의 $(s_{tB},a_{tB},r_{tB},s_{tB+1})$ 전이 전체에 대해
- 예측을 가져온다 : $Q(s_{tB},a_{tB})$
- 타깃을 가져온다 : $R(s_{tB},a_{tB})+\gamma \max_{a} (Q(s_{tB+1}, a))$
- 전체 배치 B에서 예측과 타깃 사이의 손실을 계산한다.
- 경사 하강법을 통해 손실 오차를 줄이기 위해 가중치를 업데이트한다.

<br>

### **Cartpole 예제 환경**

<br>

![](https://inspaceai.github.io/images/lhh/cartpole_rl_compare/cartpole.gif)

**1. 상태(state)**
- 카트의 위치
- 카트의 속도
- 막대기의 각도
- 막대기의 각속도

**2. 행동(action)**
- 매 스텝마다 0,1의 값을 통해 카트를 좌, 우로 조종할 수 있다.

**3. 보상(reward)**
- 매 스텝마다 카트가 중심을 기준으로 일정 범위 안에 있고, 막대기가 넘어지지 않으면 +1의 보상을 받는다.

<br>

### **Carpole 예제 코드 알고리즘**
**1. DQN 신경망 모델 정의 : class DQN(tf.keras.Model)**
- 'DQN' 클래스를 정의하여 상태를 입력으로 받고, 가능한 각 행동에 대한 Q-값을 출력하는 신경망을 생성한다.

**2. DQN 에이전트 정의 : class DQNAgent**
- **def update_target_model** : 타겟 모델을 모델의 가중치로 업데이트하는 함수 
- **def get_action** : 입실론 탐욕 정책으로 행동 선택
- **def append_sample** : 리플레이 메모리에 저장 (s,a,r,s',done)
- **def train_model** : 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
- **리플레이 메모리에 저장된 형태**

<br>

```python
def append_sample(self, state, action, reward, next_state, done):
  self.memory.append((state, action, reward, next_state, done))
```
  
<br>

![](sample.jpg)

<br>

![](sample_element.jpg)


```python
mini_batch = random.sample(self.memory, self.batch_size)

states = np.array([sample[0][0] for sample in mini_batch])
actions = np.array([sample[1] for sample in mini_batch])
rewards = np.array([sample[2] for sample in mini_batch])
next_states = np.array([sample[3][0] for sample in mini_batch])
dones = np.array([sample[4] for sample in mini_batch])
```

**3. 학습 과정**
- 각 에피소드마다 환경을 초기화하고, 에피소드가 끝날 때까지 다음을 반복한다.
    - 에이전트로부터 현재 상태에 기반한 행동을 선택 받는다.
    - 선택한 행동을 환경에 적용하고, 새로운 상태, 보상, 에피소드 종료 여부를 받는다.
    - 이 경험을 리플레이 메모리에 저장한다.
    - 메모리에 충분한 샘플이 쌓이면, 무작위로 샘플을 추출하여 에이전트를 학습시킨다.
