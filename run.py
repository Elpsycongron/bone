import math
import os
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
_MODELS = _ROOT / "models"
_DET_WEIGHTS = _MODELS / "detect_21_best.pt"
_CLS_ROOT = _MODELS / "cls_train_data"


def _default_det_weights() -> Path:
    env = os.environ.get("BONE_DET_MODEL")
    if env:
        return Path(env)
    return _DET_WEIGHTS


SCORE = {
    'Radius': [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
    'Ulna': [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
    'MCPFirst': [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
    'MCPThird': [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
    'MCPFifth': [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
    'PIPFirst': [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
    'PIPThird': [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
    'PIPFifth': [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
    'MIPThird': [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
    'MIPFifth': [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
    'DIPFirst': [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
    'DIPThird': [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
    'DIPFifth': [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49]
}

# 7 个检测类别 id（你已经在用的）
CLS = {
    "Radius": 0,
    "Ulna": 1,
    "MCPFirst": 2,
    "MCP": 3,
    "ProximalPhalanx": 4,
    "MiddlePhalanx": 5,
    "DistalPhalanx": 6,
}

# 13 个关节的保存顺序 & 名字
NAMES_13 = [
    "MCP5", "MCP3",
    "MIP5", "MIP3",
    "DIP5", "DIP3", "DIP1",
    "PIP5", "PIP3", "PIP1",
    "MCP1",
    "Ulna",
    "Radius"
]
from ultralytics import YOLO


def _cls_w(name: str) -> str:
    env = os.environ.get(f"BONE_CLS_{name.upper()}")
    return env if env else str(_CLS_ROOT / f"train_{name}" / "weights" / "best.pt")


model_DIP = YOLO(_cls_w("DIP"))
model_DIPFirst = YOLO(_cls_w("DIPFirst"))
model_MCP = YOLO(_cls_w("MCP"))
model_MCPFirst = YOLO(_cls_w("MCPFirst"))
model_MIP = YOLO(_cls_w("MIP"))
model_PIP = YOLO(_cls_w("PIP"))
model_PIPFirst = YOLO(_cls_w("PIPFirst"))
model_Radius = YOLO(_cls_w("Radius"))
model_Ulna = YOLO(_cls_w("Ulna"))

def crop_13_joints_to_memory(img_path, boxes):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    boxes = boxes.cpu().numpy()
    def pick(cls_id, idxs):
        b = boxes[boxes[:, 5] == cls_id]
        b = b[np.argsort((b[:, 0] + b[:, 2]) / 2)]  # 按x排序
        return b[idxs]

    selected = np.concatenate([
        pick(CLS["DistalPhalanx"],   [0, 2]),
        pick(CLS["MiddlePhalanx"],   [0, 2]),
        pick(CLS["ProximalPhalanx"], [0, 2, 4]),
        pick(CLS["MCP"],             [0, 2, 4]),
        pick(CLS["MCPFirst"],        [0]),
        pick(CLS["Ulna"],            [0]),
        pick(CLS["Radius"],          [0]),
    ])
    crops = {}
    for name, box in zip(NAMES_13, selected):
        x1, y1, x2, y2 = map(int, box[:4])
        crops[name] = img[y1:y2, x1:x2]
    return crops  # 获取13个小关节的位置


def to_3ch_uint8(img: np.ndarray) -> np.ndarray:
    # 让内存输入尽量等价于“从文件读入”
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # (H,W) -> (H,W,3)
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img

def predict_grade(model, crop: np.ndarray) -> int:
    crop = to_3ch_uint8(crop)
    r = model.predict(crop, verbose=False)[0]
    # 完全对齐你原始代码：names[top1] 得到的是字符串标签，比如 '7'
    grade_str = r.names[r.probs.top1]
    return int(grade_str)


def predict_bone_age(img_path):
    # 1. 目标检测（models/detect_21_best.pt，见 _default_det_weights）
    det_model = YOLO(str(_default_det_weights()))
    det_results = det_model(img_path)
    boxes21 = det_results[0].boxes.data

    # 2. 裁剪13个关节（内存）
    crops = crop_13_joints_to_memory(img_path, boxes21)

    # 3. 分类预测
    grade_DIP3 = predict_grade(model_DIP, crops["DIP3"])
    grade_DIP5 = predict_grade(model_DIP, crops["DIP5"])
    grade_DIP1 = predict_grade(model_DIPFirst, crops["DIP1"])

    grade_MCP3 = predict_grade(model_MCP, crops["MCP3"])
    grade_MCP5 = predict_grade(model_MCP, crops["MCP5"])
    grade_MCP1 = predict_grade(model_MCPFirst, crops["MCP1"])

    # MIP
    grade_MIP3 = predict_grade(model_MIP, crops["MIP3"])
    grade_MIP5 = predict_grade(model_MIP, crops["MIP5"])

    # PIP
    grade_PIP3 = predict_grade(model_PIP, crops["PIP3"])
    grade_PIP5 = predict_grade(model_PIP, crops["PIP5"])
    grade_PIP1 = predict_grade(model_PIPFirst, crops["PIP1"])

    # Radius / Ulna
    grade_Radius = predict_grade(model_Radius, crops["Radius"])
    grade_Ulna = predict_grade(model_Ulna, crops["Ulna"])

    # 4. 查表算分
    score = (
        SCORE['Radius'][grade_Radius] +
        SCORE['Ulna'][grade_Ulna] +
        SCORE['MCPFirst'][grade_MCP1] +
        SCORE['MCPThird'][grade_MCP3] +
        SCORE['MCPFifth'][grade_MCP5] +
        SCORE['PIPFirst'][grade_PIP1] +
        SCORE['PIPThird'][grade_PIP3] +
        SCORE['PIPFifth'][grade_PIP5] +
        SCORE['MIPThird'][grade_MIP3] +
        SCORE['MIPFifth'][grade_MIP5] +
        SCORE['DIPFirst'][grade_DIP1] +
        SCORE['DIPThird'][grade_DIP3] +
        SCORE['DIPFifth'][grade_DIP5]
    )

    # 5. 多项式年龄
    boneAge = 5.81191794824917 + (-0.271546561737745) * score + \
              math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
              math.pow(score, 4) * (2.0858722025667E-07) + math.pow(score, 5) * (-6.21879866563429E-10) + \
              math.pow(score, 6) * (1.19909931745368E-12) + math.pow(score, 7) * (-1.49462900826936E-15) + \
              math.pow(score, 8) * (1.162435538672E-18) + math.pow(score, 9) * (-5.12713017846218E-22) + \
              math.pow(score, 10) * (9.78989966891478E-26)

    return round(boneAge, 2)

if __name__ == '__main__':
    img_path = r'E:\PycharmProjects\project\bone\image\14782.png' # 输入图片路径
    age = predict_bone_age(img_path)
    print('这个骨骼的年龄是：',age)
