import pynvml
from pynvml import *
import torch, gc

def get_least_used_gpu():
    if torch.cuda.is_available():
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        best_gpu = 0
        min_mem = float('inf')
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            mem = meminfo.used
            if mem < min_mem:
                min_mem = mem
                best_gpu = i
        nvmlShutdown()
        return torch.device(f'cuda:{best_gpu}')
    else:
        return torch.device('cpu')