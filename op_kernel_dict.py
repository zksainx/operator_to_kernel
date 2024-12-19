import pandas as pd
import os

# 示例调用
# Type = 'forward'
# model_name='resnet18'
# op_kernel = get_op_kernel(Type, model_name)

def get_op_kernel(Type, model_name):
    dir_name = 'log_csv/' + model_name + '_' + Type

    cpu = pd.read_csv(os.path.join(dir_name, 'cpu_op.csv'))
    cuda = pd.read_csv(os.path.join(dir_name, 'cuda_runtime.csv'))
    kernel = pd.read_csv(os.path.join(dir_name, 'kernel.csv'))
    cuda = cuda[cuda['name'].str.contains('cudaLaunchKernel')].reset_index(drop=True)

    # kernel_list = set(kernel['name'].to_list())
    # print(len(kernel_list))
    # 提取cuda里的correlation和External id 两列
    cuda = cuda[['correlation', 'External id']]
    op_kernel = {}
    for i in range(len(cuda)):
        corr = cuda.loc[i, 'correlation']
        ex_id = cuda.loc[i, 'External id']
        kernel_name = kernel[kernel['correlation'] == corr]['name'].values[0]
        op_name = cpu[cpu['External id'] == ex_id]['name'].values[0]
        if op_name not in op_kernel:
            op_kernel[op_name] = set()
        op_kernel[op_name].add(kernel_name)

    return op_kernel