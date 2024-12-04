import BAC0
import inspect
import asyncio
import random

#print(inspect.signature(BAC0.core.devices.Device.Device))



async def simulate_temperature(device):
    """
    模拟温度传感器，定期更新温度值。
    """
    while True:
        new_temp = random.uniform(18, 28)  # 生成随机温度值
        device.object_list[0]["value"] = new_temp  # 更新模拟设备的温度
        print(f"[Temperature Sensor] New temperature: {new_temp:.2f}°C")
        await asyncio.sleep(3)  # 每3秒更新一次温度

async def main():
    # 启动 BACnet 客户端
    bacnet = BAC0.connect(ip="127.0.0.1")
    print("BACnet client started on localhost.")

    # 模拟设备：温度传感器
    temp_sensor =  BAC0.device(        #await
        address="127.0.0.1",  # 必需参数
        device_id=12345,      # 必需参数
        network=None,         # 通常为 None
        object_list=[
            {"type": "analogInput", "instance": 1, "value": 22.0},  # 初始模拟温度
        ]
    )
    print("Temperature Sensor created!")

    # 启动模拟任务
    await simulate_temperature(temp_sensor)

    # 停止 BACnet 客户端
    bacnet.disconnect()
    print("BACnet client disconnected.")

# 启动主事件循环
asyncio.run(main())
