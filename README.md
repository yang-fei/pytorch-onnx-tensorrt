# pytorch-onnx-tensorrt

My environment:
ubuntu 18.04
cuda10.2
cudnn8.0 (or cudnn 8.2 is ok)


Installation

pip install onnx
pip install onnxruntim
pip install pycuda

Install tensorrt version (>=8.0.1.6). Instaling instruction refer to the official website. 
1) download the tar. file and extract
2) cd Python 
3) pip install ***.whl

## Transfrom pytorch pt/pth to onnx
refer example file: change_to_onnx.py

## Transform onnx file to tensorrt file
refer example file: deploy_tensorrt.py

