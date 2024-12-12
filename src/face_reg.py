import cv2

from .build_onnx_engine import OnnxBaseModel
from .face_assign import FaceAssign
import numpy as np

mean = np.array([[[0.5, 0.5, 0.5]]])
std = np.array([[[0.5, 0.5, 0.5]]])


def process_reg_img(im):
    im = im / 255.0
    im = (im - mean) / std
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32)[None, :, :, :]
    return im


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity  # 值越大越相似


class FaceReg:
    def __init__(self, reg_model_path, assign_model_path, face_conf=0.35):
        self.model = OnnxBaseModel(reg_model_path)
        self.face_conf = face_conf
        self.assign_model = FaceAssign(assign_model_path)

    def __call__(self, key_face: str, q_face: str):
        """
        Args:
            key_face(str): 人脸模板，即想要识别的那个人的人脸图地址
            q_face(str): 待识别人脸
        Return:
            True表示人脸对比通过，否则为失败
        """
        key_face_im = cv2.imread(key_face)
        q_face_im = cv2.imread(q_face)
        assign_key_im = self.assign_model(key_face_im)
        assign_q_im = self.assign_model(q_face_im)
        process_key_im = process_reg_img(assign_key_im)
        process_q_im = process_reg_img(assign_q_im)
        key_vector = self.model.get_ort_inference(process_key_im)[0]
        q_vector = self.model.get_ort_inference(process_q_im)[0]
        score = cosine_similarity(key_vector, q_vector)
        if score > self.face_conf:
            print(f"是同一个人, 人脸分数为:{score:.2f}")
            return True
        else:
            print(f'不是同一个人, 人脸分数为:{score:.2f}')
            return False
