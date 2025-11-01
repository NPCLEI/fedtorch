import psutil
import time
import sys
import io

from datetime import datetime
from threading import Thread,Lock

def curTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

flush_counter = 0
log_cache = []

def reporttitle(idf = 'npc'):
    mem = psutil.virtual_memory()
    tmp = "%s[ %s (memo:%2.3f)] "%(curTime(),idf,mem.used / (1024**3))

    return tmp

def npclog(*inputs,idf = "",end = "\n",title = 'npc report'):

    if title:
        mem = psutil.virtual_memory()
        tmp = "%s[%s %s(memo:%2.3f)] "%(idf,curTime(),title,mem.used / (1024**3))

    ipts = ' '.join(inputs)

    log_cache.append("%s %s%s"%(tmp,ipts,end))

def logger():
    while True:
        while len(log_cache) != 0:
            print(log_cache.pop(0),end='',flush=True)
        else:
            time.sleep(0.5)

class DualOutput:
    def __init__(self):
        # 保存原始的 sys.stdout
        self.original_stdout = sys.stdout
        # 创建一个 StringIO 对象来捕捉输出
        self.captured_output = io.StringIO()
        sys.stdout = self

    def write(self, message):
        # 将消息写入 StringIO（捕捉输出）
        self.captured_output.write(message)
        # 同时写入原始 stdout（控制台）
        self.original_stdout.write(message)

    def flush(self):
        # 确保控制台输出流被刷新
        self.original_stdout.flush()

    def get_output(self):
        # 返回捕获的内容
        return self.captured_output.getvalue()

    def save_output(self, filename):
        # 将捕获的输出保存到文件
        with open(filename, 'w') as f:
            f.write(self.get_output())

    def close(self):
        sys.stdout = self.original_stdout