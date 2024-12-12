from src.face_reg import FaceReg
import os
import shutil

face_reg = FaceReg("weights/face_reg.onnx",
                   "weights/face_assign.onnx",
                   0.35)
for name in os.listdir("demo"):
    if os.path.isdir(os.path.join("demo", name)):
        for im_name in os.listdir(os.path.join("demo", name)):
            if face_reg('demo/lyf/img.png', os.path.join("demo", name, im_name)):
                shutil.copy(os.path.join("demo", name, im_name), os.path.join("out", im_name))
