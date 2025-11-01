# grid_search:



import itertools

# 定义参数
client_range = [10, 100, 1000]
pk = [0.1, 0.5, 0.8]
epoch = [1, 5, 10]
alpha = [0.01, 0.1, 1]

# 生成所有参数组合
combinations = list(itertools.product(client_range, pk, epoch, alpha))

# 打印结果
for i,combo in enumerate(combinations):
    print(i,combo)
