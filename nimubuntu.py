import os
import socket
import subprocess
from enum import Enum
from pathlib import Path
import time
import re
import atexit
from typing import List, Dict
import json

TIME_OUT = 1800

class ModelType(Enum):
    FLUX_DEV = "FLUX_DEV"
    FLUX_CANNY = "FLUX_CANNY"
    FLUX_DEPTH = "FLUX_DEPTH"
    FLUX_SCHNELL = "FLUX_SCHNELL"


# --- 新的 Ubuntu/Docker 管理类 ---
class NIMManager_ubuntu:
    
    MODEL_REGISTRY: Dict[ModelType, str] = {
        ModelType.FLUX_DEV: "ewr.vultrcr.com/3y2025/nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_CANNY: "ewr.vultrcr.com/3y2025/nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_DEPTH: "ewr.vultrcr.com/3y2025/nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_SCHNELL: "ewr.vultrcr.com/3y2025/nvcr.io/nim/black-forest-labs/flux.1-schnell:1.0.0",
    }
    PORT = 8000

    def __init__(self):
        self._nim_server_proc_dict: Dict[ModelType, Dict] = {}
        self.api_key = os.environ.get("NGC_API_KEY", "")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.local_nim_cache   = os.environ.get("LOCAL_NIM_CACHE", "")
        
    def _run_cmd(self, cmd: str, err_msg: str = "Unknown") -> List[str]:
        
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True, text=True)
        # check=True 会在返回码非0时自动抛出 CalledProcessError
        return result.stdout.splitlines()

    def _run_proc(self, cmd: str):
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

    def _setup_directories(self, model_name: ModelType) -> None:
        from pathlib import Path
        cache_path = self.local_nim_cache
        if not cache_path:
            raise Exception("LOCAL_NIM_CACHE 环境变量未设置")
        
    def get_running_container_info(self) -> Dict:
        
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
        containers_data = self.get_running_container_info()
        if model_name.value in containers_data:
            if model_name in self._nim_server_proc_dict.keys():
                print(f"Found container  : {model_name.value}, Running on port {containers_data[model_name.value]['ports'][0]}")
                return True
        return False


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

 
    def get_port(self, model_name: ModelType) -> int:
        # ... (逻辑不变)
        if model_name in self._nim_server_proc_dict:
            return self._nim_server_proc_dict[model_name]["port"]
        containers_data = self.get_running_container_info()
        if model_name.value in containers_data:
            return containers_data[model_name.value]["ports"][0]
        raise Exception(f"NIM {model_name.value} is not running.")
 


# --- 测试入口 (更新为新类名) ---
if __name__ == "__main__": 

    model_name = ModelType.FLUX_DEV 
    manager = NIMManager_ubuntu()
    try:
        manager.is_nim_running(model_name)
        print(f"NIM for {model_name.value} is running on port {manager.get_port(model_name)}")
        print("Waiting for 20 seconds  ...")
        time.sleep(20)
    finally:
        print("OK.")
