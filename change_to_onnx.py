#--*-- coding:utf-8 --*--
import onnx 
import torch
import torchvision 
from torch import nn
# import netron

import torch.nn.functional as F
class myHardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


from models.experimental import attempt_load
from utils.torch_utils import select_device

gpu = str(0)
device = select_device('cpu')
print(device)

weights="20210401_zhikong_640.pt"
model = attempt_load(weights, map_location=device)  # load FP32 model
# net.half()  # to FP16

for k, m in model.named_modules():
    try:
        if isinstance(m.act, nn.Hardswish):
            print(m)
            m.act = myHardswish()
    except:
        continue

export_onnx_file = "./yolo5.onnx"
x=torch.onnx.export(model,  # 待转换的网络模型和参数
                torch.randn(1, 3, 640, 640, device='cpu'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                export_onnx_file,  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["input"], # + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=12,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                # dynamic_axes={"input":{0: "batch_size" }, "output":{0: "batch_size"},} #设置动态维度，此处指明input节点的第0维度可变，命名为batch_size
                )

# import onnx  # 注意这里导入onnx时必须在torch导入之前，否则会出现segmentation fault
net = onnx.load("./yolo5.onnx")  # 加载onnx 计算图
onnx.checker.check_model(net)  # 检查文件模型是否正确
onnx.helper.printable_graph(net.graph)  # 输出onnx的计算图

import onnxruntime
import numpy as np

# netron.start("./yolo4tr.onnx")

session = onnxruntime.InferenceSession("./yolo5.onnx") # 创建一个运行session，类似于tensorflow
out_r = session.run(None, {"input": np.random.rand(1, 3, 640, 640).astype('float32')})  # 模型运行，注意这里的输入必须是numpy类型
print(len(out_r))
print(out_r[0].shape)




