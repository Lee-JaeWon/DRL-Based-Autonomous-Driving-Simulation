# submission후 evaluation이 진행 되는 과정을 보여주는 code입니다.
# 채점 서버에서 다음과 같이 제출한 python script로 부터
# Agent class를 상속받고, load_model()로 제출한 model을 불러온 후
# 채점이 시작됩니다.

from DQN_Gym import DQN_Gym
import numpy as np
from DQN_CAR import Agent

def main():
    env = DQN_Gym(
            time_scale=1,
            port=11000, width=1600, height=900,
            filename='E:/RL_Algorithm/Autonomous_Driving_DRL/ML_test(ver9_fullsensor)/ML_test.exe')

    score = 0
    episode_step = 0
    episode = 1000

    agent = Agent() # user가 제출한 Agent class에서 불러오기.
    try:
        agent.load_model() # user의 모델 불러오기. 경로는 best_model 폴더.
        print("Succeed load User's model");print()
    except:
        print("Fail load User's model'")
        raise
    
    print("Model Inference Start")
    print("-------------------------------------------");print()
    for epi in range(episode):
        
        state = env.reset()
        while True:

            action = agent.policy(state)
            next_state, reward, done = env.step(action)

            score += reward
            episode_step += 1
            state = next_state

            if done:
                break
        print('episode: ', epi, ' score: ', score, ' everage score: ', score/episode_step)
        score = 0
        episode_step = 0
    

if __name__ == '__main__':
    main()