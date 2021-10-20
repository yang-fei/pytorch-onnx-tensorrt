# pytorch-onnx-tensorrt

My environment:
ubuntu 18.04
cuda10.2
cudnn8.0 (or cudnn 8.2 is ok)
sucessfully transform pytorch-yolov5 to tensorrt. And the reference time is reduced into half. 


Installation

pip install onnx
pip install onnxruntim
pip install pycuda

Install tensorrt version (>=8.0.1.6). Instaling instruction refer to the official website. 
1) download the tar. file and extract
2) cd Python 
3) pip install ***.whl

## Transform pytorch pt/pth to onnx
refer example file: change_to_onnx.py

## Transform onnx file to tensorrt file
refer example file: deploy_tensorrt.py

# Remarks
1. Tensorrt version < 8.0.1.6 cannot support scatterND plugins. I found no ways to resovle the problem in two days. The best way I think is to re-install tensorrt with a higher version. 
2. Jetpack >=4.6 will contains the tensorrt 8.0.1. I failed to upgrade the tensorrt version on Jetson NX. The best way is to re-install the jetson system with Jetpack4.6.  
3. Tensort 7.1.* onnx_to_tensorrt example is different from that of tensorrt 8.*. 










