import os
import socket
import subprocess
from enum import Enum
from pathlib import Path
from .ngc import get_ngc_key  # 假设此函数不变
import time
import re
import atexit
from typing import List, Dict
import json


# --- 常量和枚举 (保持不变) ---
TIME_OUT = 1800

class ModelType(Enum):
    FLUX_DEV = "FLUX_DEV"
    FLUX_CANNY = "FLUX_CANNY"
    FLUX_DEPTH = "FLUX_DEPTH"
    FLUX_SCHNELL = "FLUX_SCHNELL"

class OffloadingPolicy(Enum):
    NONE = "None"
    SYS = "System RAM"
    DISK = "Disk"
    DEFAULT = "Default"

# --- 新的 Ubuntu/Docker 管理类 ---
class NIMManager_ubuntu:
    '''
    此类负责在 Ubuntu 环境下使用 Docker 管理 NIM 容器。
    '''

    # 模型注册表 (保持不变)
    MODEL_REGISTRY: Dict[ModelType, str] = {
        ModelType.FLUX_DEV: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_CANNY: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_DEPTH: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_SCHNELL: "nvcr.io/nim/black-forest-labs/flux.1-schnell:1.0.0",
    }
    PORT = 5000

    # --- 4.1. 初始化与环境设置 ---
    def __init__(self):
        """
        初始化管理器。移除所有WSL相关逻辑。
        """
        self._nim_server_proc_dict: Dict[ModelType, Dict] = {}
        self.api_key = get_ngc_key()
        self.cache_path_template = self._get_cache_path_template()
        atexit.register(self.cleanup)
        # 不再需要 cmd_prefix

    def _get_cache_path_template(self) -> str:
        """
        获取缓存路径模板。逻辑简化为纯 Linux 环境。
        """
        home = Path.home()
        # {model_name} 是一个占位符，将在使用时格式化
        cache_path = home / "nimcache/{model_name}/latest/.cache"
        return cache_path.as_posix()

    def _setup_directories(self, model_name: ModelType) -> None:
        """
        创建 NIM 缓存所需的目录。
        """
        cache_path = self.cache_path_template.format(model_name=model_name.value)
        if os.path.exists(cache_path):
            return
        os.makedirs(cache_path, exist_ok=True)
        # Linux下通常不需要强制修改权限，但如果需要，可以保留
        # chmod_command = f"chmod 777 -R {Path(cache_path).parent}"
        # self._run_cmd(chmod_command, "setup cache directory")
        print(f"Directory setup completed for {model_name.value} at {cache_path}")

    # --- 4.2. 命令执行 (简化) ---
    def _run_cmd(self, cmd: str, err_msg: str = "Unknown") -> List[str]:
        """
        执行同步命令，移除 cmd_prefix。
        """
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True, text=True)
        # check=True 会在返回码非0时自动抛出 CalledProcessError
        return result.stdout.splitlines()

    def _run_proc(self, cmd: str):
        """
        执行异步后台进程，移除 cmd_prefix。
        """
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

    # --- 4.3. 核心 Docker 操作 ---
    def pull_nim_image(self, model_name: ModelType) -> None:
        """
        使用 docker 命令拉取 NIM 镜像。
        """
        # 使用 password-stdin 的方式更安全地登录
        login_command = f"echo \"{self.api_key}\" | docker login nvcr.io --username '$oauthtoken' --password-stdin"
        self._run_cmd(login_command, "Docker login to nvcr.io")

        registry_path = self.MODEL_REGISTRY[model_name]
        pull_command = f"docker pull {registry_path}"
        
        # 实时输出拉取进度的逻辑保持不变
        process = self._run_proc(pull_command)
        # ... (此处省略与原代码相同的日志读取循环)
        # 等待进程结束
        process.wait()
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise Exception(f"Failed to pull the image: {stderr}")
        print("Image has been pulled successfully.")


    def get_running_container_info(self) -> Dict:
        """
        获取正在运行的 Docker 容器信息，并适配 docker ps 的输出。
        """
        cmd = 'docker ps -a --format "{{json .}}"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error fetching Docker containers:", result.stderr)
                return {}

            containers_data = {}
            for line in result.stdout.strip().splitlines():
                if not line: continue
                container_json = json.loads(line)
                name = container_json.get("Names")
                # Docker 的端口格式是 "0.0.0.0:5000->8000/tcp, :::5000->8000/tcp"
                # 我们需要从中解析出主机端口
                port_str = container_json.get("Ports", "")
                host_port = None
                match = re.search(r'0\.0\.0\.0:(\d+)->8000/tcp', port_str)
                if match:
                    host_port = int(match.group(1))

                if name and host_port:
                    containers_data[name] = {
                        "ports": [host_port],
                        "id": container_json.get("ID"),
                        "image": container_json.get("Image")
                    }
            return containers_data
        except Exception as e:
            print(f"Failed to parse docker container info: {e}")
            return {}

    def is_nim_running(self, model_name: ModelType) -> bool:
        """
        检查指定模型的 NIM 容器是否正在运行。逻辑基本不变。
        """
        # ... (此方法逻辑与原版相同，依赖 get_running_container_info)
        containers_data = self.get_running_container_info()
        if model_name.value in containers_data:
            if model_name in self._nim_server_proc_dict.keys():
                return True
            # 如果存在孤儿容器，先停止它
            print(f"Found an orphaned container for {model_name.value}. Stopping it before starting a new one.")
            self.stop_nim(model_name, force=True)
        return False
        
    def start_nim_container(self, model_name: ModelType, offloading_policy: OffloadingPolicy, hf_token: str = "") -> None:
        """
        使用 docker run 启动 NIM 容器。
        """
        if self.is_nim_running(model_name):
            print(f"NIM for {model_name.value} is already running...")
            return

        self._setup_directories(model_name)
        cache_path = self.cache_path_template.format(model_name=model_name.value)

        # 端口分配逻辑不变
        port = self.PORT + len(self._nim_server_proc_dict)
        while self.is_port_in_use(port):
            port += 1
        
        # 模型变体逻辑不变
        variant = self._get_variant(model_name)

        # 构建 docker run 命令
        command = (
            f"docker run --rm "
            f"--gpus all " # 使用 --gpus 标志
            f"--name={model_name.value} "
            f"--shm-size=16GB "
            f"-e NGC_API_KEY={self.api_key} "
            f"-v {cache_path}:/opt/nim/.cache "
            f"-e NIM_RELAX_MEM_CONSTRAINTS=1 "
            f"-e NIM_OFFLOADING_POLICY={offloading_policy.value.replace(' ', '_').lower()} "
            f"-e NIM_MODEL_VARIANT={variant} "
            f"-e HF_TOKEN={hf_token} "
            f"-p {port}:8000 "
            f"{self.MODEL_REGISTRY[model_name]}"
        )
        print("Executing command:", command)

        # 启动进程和健康检查的逻辑完全复用原代码
        process = self._run_proc(command)
        self._nim_server_proc_dict[model_name] = {"port": port, "id": None} # id 可以在之后从 docker ps 获取
        # ... (此处省略与原代码完全相同的健康检查、日志读取循环)

    def stop_nim(self, model_name: ModelType, force: bool = False) -> None:
        """
        使用 docker stop 停止 NIM 容器。
        """
        if not force:
             # is_nim_running 依赖 get_running_container_info，所以这里要先检查外部容器
            containers = self.get_running_container_info()
            if model_name.value not in containers:
                 print(f"NIM container {model_name.value} is not running.")
                 if model_name in self._nim_server_proc_dict:
                    self._nim_server_proc_dict.pop(model_name)
                 return

        command = f"docker stop {model_name.value}"
        try:
            self._run_cmd(command, f"stop NIM {model_name.value}")
        except subprocess.CalledProcessError as e:
            # 如果容器已经不存在，docker stop会报错，可以忽略
            if "No such container" in e.stderr:
                print(f"Container {model_name.value} was already stopped or removed.")
            else:
                raise e # 重新抛出其他错误

        if model_name in self._nim_server_proc_dict:
            self._nim_server_proc_dict.pop(model_name)
        print(f"Stopped NIM {model_name.value}")


    # --- 4.4. 辅助与清理方法 (部分修改) ---
    def deploy_nim(self, model_name: ModelType, offloading_policy: OffloadingPolicy, hf_token: str) -> None:
        """
        部署 NIM 的高级接口。逻辑不变。
        """
        self.pull_nim_image(model_name)
        self.start_nim_container(
            model_name,
            offloading_policy,
            hf_token
        )

    def cleanup(self) -> None:
        """
        脚本退出时自动清理由本实例启动的容器。
        """
        nims_to_stop = list(self._nim_server_proc_dict.keys())
        print(f"Cleaning up managed NIM containers: {[m.value for m in nims_to_stop]}")
        for model in nims_to_stop:
            try:
                # 使用 docker stop 命令
                command = f"docker stop {model.value}"
                subprocess.run(command, shell=True, check=False, capture_output=True)
                print(f"Stopping NIM {model.value}")
            except Exception as e:
                print(f"Error during cleanup for {model.value}: {e}")

    # --- 以下方法基本保持不变 ---
    def _get_variant(self, model_name: ModelType) -> str:
        # ... (逻辑不变)
        if model_name.value.endswith("CANNY"):
            return "canny"
        elif model_name.value.endswith("DEPTH"):
            return "depth"
        elif model_name.value.endswith("SCHNELL"):
            return "base"
        else:
            return "base"

    def is_port_in_use(self, port: int) -> bool:
        # ... (逻辑不变)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def get_port(self, model_name: ModelType) -> int:
        # ... (逻辑不变)
        if model_name in self._nim_server_proc_dict:
            return self._nim_server_proc_dict[model_name]["port"]
        containers_data = self.get_running_container_info()
        if model_name.value in containers_data:
            return containers_data[model_name.value]["ports"][0]
        raise Exception(f"NIM {model_name.value} is not running.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()


# --- 测试入口 (更新为新类名) ---
if __name__ == "__main__":
    # 假设 get_ngc_key() 和 hf_token 可用
    # from some_secrets_manager import get_hf_token

    model_name = ModelType.FLUX_DEV
    offloading_policy = OffloadingPolicy.DEFAULT
    # hf_token = get_hf_token()

    manager = NIMManager_ubuntu()
    try:
        manager.deploy_nim(model_name, offloading_policy, hf_token="")
        print(f"NIM for {model_name.value} is running on port {manager.get_port(model_name)}")
        print("Waiting for 20 seconds before stopping...")
        time.sleep(20)
    finally:
        manager.stop_nim(model_name)
        print("Cleanup complete.")