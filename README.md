1. test_op_kernel.ipynb 演示VGG16,resnet18, LSTM, transformer 模型的op-kernel映射 
2. pytorch_tracing.py  用pytorch profiler在模型的forward，backward，optimize阶段进行tracing
3. op_kernel_dict.py 从log_csv提取模型在单个阶段中op和kernel的映射关系