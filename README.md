10.15
1、替换其他大模型（本文用GPT-3.5，可扩展其他主流模型）
https://github.com/chatchat-space/Langchain-Chatchat****
https://github.com/langchain-ai/chat-langchain**
2、替换scenario（本文是highway场景，可扩展到Merge、roundabout、parking、intersection、racetrack ）
wordmodel、
3、不同prompt strategy（本文应用的ReAct strategy，可扩展到其他策略例如:AISOD等）

4、Metric（需要对实验结果进行定性定量分析，从safe (collision rate) ，efficient (average speed) ，comfort (jerk)，social-impact (mean acc of other vehicles) 、driver-style (?)  hallucination , time delay (from obs->action)等角度找到合适的评价标准）

5、其他（引入 各种Coach、仿真器carla）