import sys
import platform

# 获取平台信息
platform_info = sys.platform
print("platform:", platform_info)

# 获取机器类型
machine_type = platform.machine()
print("platform machine:", machine_type)