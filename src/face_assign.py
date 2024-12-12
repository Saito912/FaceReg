import numpy as np
from skimage import transform as trans
from .build_onnx_engine import OnnxBaseModel
import cv2

mean = np.array([[[0.5, 0.5, 0.5]]])
std = np.array([[[0.5, 0.5, 0.5]]])


class FaceAssign:
    reference_ldmk = np.array([[38.29459953, 51.69630051],
                               [73.53179932, 51.50139999],
                               [56.02519989, 71.73660278],
                               [41.54930115, 92.3655014],
                               [70.72990036, 92.20410156]])

    def __init__(self, model_path):
        self.model = OnnxBaseModel(model_path)

    def __call__(self, image):
        new_face_im = cv2.resize(image.copy(), (160, 160))
        face_im = new_face_im.copy() / 255.0
        face_im = (face_im - mean) / std
        face_im = face_im.transpose((2, 0, 1))
        face_im = face_im.astype(np.float32)[None, :, :, :]
        pred = self.model.get_ort_inference(face_im)[0]
        bbox, cls_labels, ldmk = pred[:4], pred[4:6], pred[6:]
        h, w, c = new_face_im.shape
        vis_ldmk = ldmk.copy().reshape((5, 2))
        ldmk = ldmk.reshape(1, 5, 2)
        vis_ldmk[:, 0] = vis_ldmk[:, 0] * w
        vis_ldmk[:, 1] = vis_ldmk[:, 1] * h
        assign_ldmk = self.get_cv2_affine_from_landmark(ldmk)[0]
        dst = cv2.warpAffine(new_face_im, assign_ldmk, (112, 112))
        return dst

    def get_cv2_affine_from_landmark(self, ldmks):
        ldmks = ldmks * 160
        transforms = []
        for ldmk in ldmks:
            tform = trans.SimilarityTransform()
            tform.estimate(ldmk, self.reference_ldmk)
            M = tform.params[0:2, :]
            transforms.append(M)
        transforms = np.stack(transforms, axis=0)
        return transforms
