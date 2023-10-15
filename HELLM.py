import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
# import openai
# openai.api_base = "https://api.chatanywhere.com.cn/v1"
from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)

import sys

if False:
    # 打开一个文件以保存输出
    output_file = open('./results-video/log.txt', 'w')
    sys.stdout = output_file
    # ... (保持代码不变)

def main():
    #step-1 创建大模型#######################################################
    OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
        llm = AzureChatOpenAI(
            deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
            temperature=0,
            max_tokens=1024,
            request_timeout=60
        )
    elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
        # os.environ["https_proxy"] = "http://127.0.0.1:10809"
        # os.environ["http_proxy"] = "http://127.0.0.1:10809"
        llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-16k-0613', # or any other model with 8k+ context
            max_tokens=1024,
            request_timeout=60
        )#设置大模型

    #step-2 创建小车环境####################################################
    # base setting
    vehicleCount = 15
    # environment setting
    config = {
        "observation": {
            "type": "Kinematics", #表示使用运动学信息。
            "features": ["presence", "x", "y", "vx", "vy"], #"presence"（车辆存在与否）、"x"（车辆的 x 坐标）、"y"（车辆的 y 坐标）、"vx"（车辆的 x 方向速度）和 "vy"（车辆的 y 方向速度）。
            "absolute": True, #设置为 True，表示观测信息的坐标是绝对坐标（而不是相对坐标）。
            "normalize": False, #设置为 False，表示不对观测信息进行归一化处理。
            "vehicles_count": vehicleCount,
            "see_behind": True, #设置为 True，表示可以观测到车辆后面的信息。
        },
        "action": {
            "type": "DiscreteMetaAction",#表示离散的元行动。
            "target_speeds": np.linspace(0, 32, 9),# 生成一个包含9个不同速度值的数组，范围从0到32，用于表示不同的目标速度选项。
        },
        "duration": 40,#为40个时间步长。
        "vehicles_density": 2,#制仿真中车辆的分布密度。
        "show_trajectories": True,#表示显示车辆的轨迹信息。
        "render_agent": True,#表示渲染驾驶代理的视图。
    }


    #env = gym.make('highway-v0', render_mode="rgb_array")#gymnasium来创建highway场景
    env = gym.make('highway-v0', render_mode="rgb_array")
    #env = gym.make("merge-v0", render_mode="rgb_array")
    #env = gym.make("roundabout-v0", render_mode="rgb_array")
    #env = gym.make("parking-v0", render_mode="rgb_array")
    #env = gym.make("intersection-v0", render_mode = "rgb_array")
    #env = gym.make("racetrack-v0", render_mode = "rgb_array")
    env.configure(config)
    env = RecordVideo(
        env, './results-video',
        name_prefix=f"highwayv0"
    )
    env.unwrapped.set_record_video_wrapper(env)
    obs, info = env.reset()#这行代码初始化仿真环境，并获取初始的观测（observation）和信息（info）。
    #info信息：车辆速度、是否发生碰撞、采取的行动以及奖励信息等。{'speed': 25, 'crashed': False, 'action': 0, 'rewards': {'collision_reward': 0.0, 'right_lane_reward': 0.6666666666666666, 'high_speed_reward': 0.5, 'on_road_reward': 1.0}}
    #obs信息：ndarray:(15,5),仿真环境中的15个车辆状态， ["presence", "x", "y", "vx", "vy"]，
    env.render()#渲染

    # scenario and driver agent setting
    if not os.path.exists('results-db/'):
        os.mkdir('results-db')
    database = f"results-db/highwayv0.db"
    sce = Scenario(vehicleCount, database) #用于管理仿真环境中的场景和车辆
    toolModels = [
        getAvailableActions(env),
        getAvailableLanes(sce),
        getLaneInvolvedCar(sce),
        isChangeLaneConflictWithCar(sce),
        isAccelerationConflictWithCar(sce),
        isKeepSpeedConflictWithCar(sce),
        isDecelerationSafe(sce),
        isActionSafe(),
    ]#重点代码，prompt工程的修改，类似数据增强的list；用于从仿真环境中获取不同类型的信息，例如可用行动、可用车道、与其他车辆的冲突等。

    #stpe3-创建代理（需要与仿真环境中的小车进行互动）################################################
    DA = DriverAgent(llm, toolModels, sce, verbose=True)#该代理与仿真环境中的车辆进行互动，并使用工具模型来分析驾驶环境和做出决策。
    outputParser = OutputParser(sce, llm)#用于解析驾驶代理的输出并将其转换为仿真环境可以理解的形式。
    output = None
    done = truncated = False
    frame = 0

    try:#如果退出保存录像
        while not (done or truncated):
            sce.upateVehicles(obs, frame)#更新仿真场景中的车辆状态。
            DA.agentRun(output)#让代理分析环境并做出驾驶决策。
            da_output = DA.exportThoughts()
            output = outputParser.agentRun(da_output)#将驾驶代理的输出传递给 outputParser 进行解析，生成output仿真可以理解的形式。
            env.render()#渲染当前的仿真画面。
            env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
            #执行仿真环境的一步，将驾驶代理的行动应用到环境中，获取奖励、碰撞信息等，并更新观测信息 obs。
            obs, reward, done, info, _ = env.step(output["action_id"])
            print(output)#打印输出信息。
            frame += 1#递增仿真帧数 frame。
    finally:
        env.close()

    # 恢复标准输出
    sys.stdout = sys.__stdout__
    # 关闭输出文件
    output_file.close()
if __name__ == '__main__':
    main()