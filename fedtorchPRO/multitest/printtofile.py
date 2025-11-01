import sys
import io

class DualOutput:
    def __init__(self):
        # 保存原始的 sys.stdout
        self.original_stdout = sys.stdout
        # 创建一个 StringIO 对象来捕捉输出
        self.captured_output = io.StringIO()

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

# 实例化 DualOutput 对象
dual_output = DualOutput()

# 将 sys.stdout 替换为 DualOutput
sys.stdout = dual_output

# 现在的 print 会被捕捉，同时显示在控制台
print("This will be printed and captured.")
print("This is another line.")

# 获取捕获的输出
captured_output = dual_output.get_output()
print("Captured Output:\n", captured_output)

# 保存到文件
dual_output.save_output('output.txt')

# 恢复原始 stdout
sys.stdout = dual_output.original_stdout

# 继续打印内容到控制台
print("This will only appear in the console.")
