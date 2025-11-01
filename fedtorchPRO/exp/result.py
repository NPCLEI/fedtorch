import os


def extract_metrics(data_directory,lastn = 0.2):
    results = []

    # 遍历每个算法文件夹
    for algorithm in os.listdir(data_directory):
        algorithm_path = os.path.join(data_directory, algorithm)

        if os.path.isdir(algorithm_path):
            acc_file_path = os.path.join(algorithm_path, 'acc.txt')

            # 读取acc.txt文件
            if os.path.isfile(acc_file_path):
                with open(acc_file_path, 'r') as f:
                    data = eval(f.read())  # 读取并解析数据
                    rounds, metrics = zip(*data)

                    # 提取最后轮次的ACC值
                    lastn = int(lastn * len(metrics) + 1) if 0 < lastn < 1 else  lastn
                    last_metrics = metrics[-lastn:]  # 取最后5轮的数据
                    avg = sum(last_metrics) / len(last_metrics)
                    std = (sum((x - avg) ** 2 for x in last_metrics) / len(last_metrics)) ** 0.5

                    results.append({
                        'Algorithm': algorithm,
                        'avg±std': f"{avg:.4f}±{std:.4f}"
                    })

    return results

def convert_to_pandas(results):
    try:
        import pandas as pd
    except:
        pass
    
    # return results

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 找到最佳平均值
    best_index = df['avg±std'].apply(lambda x: float(x.split('±')[0])).idxmax()
    df.at[best_index, 'avg±std'] += ' (Best)'

    return df

if __name__ == '__main__':

    # 使用函数
    data_directory = './workspace/FashionMNIST/A0d1C500/A0.1C500(2024_11_04_19_58_32)'  # 替换为你的数据目录
    df = extract_metrics(data_directory)
    df = convert_to_pandas(df)

    output_path = './workspace/FashionMNIST/metrics_results.xlsx'
    df.to_excel(output_path, index=False)

    # 显示结果
    print(df)
