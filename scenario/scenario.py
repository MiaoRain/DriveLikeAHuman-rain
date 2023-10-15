from scenario.baseClass import Lane, Vehicle
from typing import List, Dict
from datetime import datetime
from rich import print
import sqlite3
import json
import os


class Scenario:
    def __init__(self, vehicleCount: int, database: str = None) -> None:
        self.lanes: Dict[str, Lane] = {}
        self.getRoadgraph()
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicleCount = vehicleCount
        self.initVehicles()

        if database:
            self.database = database
        else:
            self.database = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)

        conn = sqlite3.connect(self.database)#创建与SQLite 数据库的连接，并将连接对象赋值给变量 conn。
        cur = conn.cursor()# 创建数据库游标对象，并将其赋值给变量 cur，游标用于执行 SQL 查询和操作数据库。
        cur.execute(
            """CREATE TABLE IF NOT EXISTS vehINFO(
                frame INT,
                id TEXT,
                x REAL,
                y REAL,
                lane_id TEXT,
                speedx REAL,
                speedy REAL,
                PRIMARY KEY (frame, id));"""
        )#执行 SQL 命令，创建两个数据库表，一个用于存储车辆信息 (vehINFO)，另一个用于存储决策信息 (decisionINFO)。
        cur.execute(
            """CREATE TABLE IF NOT EXISTS decisionINFO(
                frame INT PRIMARY KEY,
                scenario TEXT,
                thoughtsAndActions TEXT,
                finalAnswer TEXT,
                outputParser TEXT);"""
        )
        conn.commit()#提交对数据库的更改。
        conn.close()#关闭数据库连接。

        self.frame = 0

    def getRoadgraph(self):#定义了 getRoadgraph 方法，用于初始化道路图，包括创建不同的道路（Lane 对象）并将它们添加到 self.lanes 字典中。
        for i in range(4):
            lid = 'lane_' + str(i)
            leftLanes = []
            rightLanes = []
            for j in range(i+1, 4):
                rightLanes.append('lane_' + str(j))
            for k in range(0, i):
                leftLanes.append('lane_' + str(k))
            self.lanes[lid] = Lane(
                id=lid, laneIdx=i,
                left_lanes=leftLanes,
                right_lanes=rightLanes
            )

    def initVehicles(self):#定义了 initVehicles 方法，用于初始化车辆，包括创建不同的车辆（Vehicle 对象）并将它们添加到 self.vehicles 字典中。
        for i in range(self.vehicleCount):
            if i == 0:
                vid = 'ego'
            else:
                vid = 'veh' + str(i)
            self.vehicles[vid] = Vehicle(id=vid)
#定义了 upateVehicles 方法，用于更新车辆信息，根据观测数据更新车辆的状态，并将车辆信息写入数据库。
    def upateVehicles(self, observation: List[List], frame: int):
        self.frame = frame
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        for i in range(len(observation)):
            if i == 0:
                vid = 'ego'
            else:
                vid = 'veh' + str(i)
            presence, x, y, vx, vy = observation[i]
            if presence:
                veh = self.vehicles[vid]
                veh.presence = True
                veh.updateProperty(x, y, vx, vy)
                cur.execute(
                    '''INSERT INTO vehINFO VALUES (?,?,?,?,?,?,?);''',
                    (frame, vid, float(x), float(y),
                     veh.lane_id, float(vx), float(vy))
                )
            else:
                self.vehicles[vid].clear()

        conn.commit()
        conn.close()

    def export2json(self):#将场景的信息导出为 JSON 格式的数据。
        scenario = {}
        scenario['lanes'] = []
        scenario['vehicles'] = []
        for lv in self.lanes.values():
            scenario['lanes'].append(lv.export2json())
        scenario['ego_info'] = self.vehicles['ego'].export2json()

        for vv in self.vehicles.values():
            if vv.presence:
                scenario['vehicles'].append(vv.export2json())

        return json.dumps(scenario)
