import subprocess
import psutil
import os
import signal

# Start CARLA server in a new process group
def start_server(port):
    #subprocess.run(
    #                f'taskset -c {int((port/1000-2)*6)}-{int((port/1000-2)*6+5)} ../carla_custom/carla_build/CarlaUE4.sh -carla-port={port} -vulkan -nosound -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={0} -RenderOffScreen -quality-level=Low &',
    #                shell=True)
    server_process = subprocess.Popen(['C:/Users/ITWM/Desktop/WindowsNoEditor/CarlaUE4.exe', '-carla-port=' + str(port),'-vulkan','-carla-server', '-quality-level=Low'])#, "-RenderOffScreen"])
    print("CARLA server is running in the background!")
#'-dx11',
# Function to kill a process and all its children
def kill():
    # Windows
    subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"])
    # Linux
    #subprocess.run(["pkill", "-9", "CarlaUE4"])
