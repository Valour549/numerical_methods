import cv2
import re
from matplotlib import pyplot as plt
import pytesseract
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile

import sys
import os

GPU = False

if GPU:
    # for GPU version
    if os.path.exists(".yolo_cpp_dll.dll"):
        os.rename(".yolo_cpp_dll.dll", "yolo_cpp_dll.dll")
else:
    # for CPU version
    if os.path.exists("yolo_cpp_dll.dll"):
        os.rename("yolo_cpp_dll.dll", ".yolo_cpp_dll.dll")
import darknet


class yolo:
    def __init__(self):
        # 讀取 model 權重檔案等
        self.network, self.class_names, self.class_colors = darknet.load_network('cfg/yolov4-tiny-obj.cfg', 'cfg/plate.data', 'weights/yolov4-tiny-obj_final.weights', batch_size = 1)

class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        # 建立 GUI，設定標題、大小
        self.root = master
        self.root.title("車牌辨識")
        self.root.geometry("900x500")
        self.pack(fill = tk.BOTH)

        # 配置按鈕的位置、指令、文字
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill = tk.BOTH, anchor = "w")
        
        self.select_button = tk.Button(self.button_frame, command = self.select_image, text = "選擇圖片")
        self.select_button.pack(side = tk.LEFT, padx = 20, pady = 10)

        self.save_button = tk.Button(self.button_frame, command = self.save_image, text = "儲存圖片")
        self.save_button.pack(side = tk.LEFT)
        
        # 配置圖片的位置
        self.image_frame = ttk.Frame(self)
        self.image_frame.pack(fill = tk.BOTH, anchor = "w")
        
        self.original_image_canvas = tk.Canvas(self.image_frame, width = 400, height = 400, bg = "white")
        self.original_image_canvas.pack(side = tk.LEFT, padx = 20)
        
        self.result_image_canvas = tk.Canvas(self.image_frame, width = 400, height = 400, bg = "white")
        self.result_image_canvas.pack(side = tk.LEFT, padx = 20)
        
        # 讀取 model
        self.model = yolo()

        # 設定文字辨識的執行檔位置
        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"
    
    # 選擇圖片的按鈕
    def select_image(self):
        # 可選擇的檔案種類
        filetypes = (
            ('Image', '*.jpg'),
            ('Image', '*.png'),
            ('All files', '*.*')
        )
        # 發出選擇檔案的請求
        filename = askopenfilename(filetype = filetypes)
        # 如果選擇檔案存在則讀取檔案、顯示在左方畫面上、進行偵測並將結果顯示在右方螢幕上
        if filename:
            self.cv2_image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.image = Image.open(filename)
            self.img = ImageTk.PhotoImage(self.image.resize((400, 400)))
            self.original_image_canvas.create_image(0, 0, anchor = "nw", image = self.img)
            self.detect(self.cv2_image)

    # 儲存結果圖片的按鈕
    def save_image(self):
        try:
            # 如果有結果才會發出選擇請求
            if self.pillow_image:
                # 檔案種類
                filetypes = (
                    ('Image', '*.jpg'),
                    ('Image', '*.png'),
                    ('All files', '*.*')
                )
                # 選擇檔案儲存位置
                filestream = asksaveasfile(initialfile = "output.jpg", filetype = filetypes)
                if filestream:
                    # 將圖片保存在指定位置
                    self.pillow_image.save(filestream.name)
                    filestream.close()
        except:
            pass
    
    # 進行車牌偵測
    def detect(self, image):
        # 取得圖片長寬
        h, w = image.shape[:2]
        # 將圖片轉成常見的RGB格式並調整大小
        resized_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h))
        detections_results = []
        
        # 將圖片轉成 Bytes 形式讓模型讀取
        darknet_image = darknet.make_image(w, h, 3)
        darknet.copy_image_from_bytes(darknet_image, resized_image.tobytes())

        # 將圖片餵入模型，設定 threshold，取得偵測結果
        detections = darknet.detect_image(self.model.network, self.model.class_names, darknet_image, thresh = 0.1)
        
        for label, confidence, bbox in detections:
            # 將結果範圍轉換成四個點座標
            xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
            if xmin >= 0 and ymin >= 0 and xmax >= 0 and ymax >= 0:
                # 將車牌位置圖片取出以餵入OCR文字辨識
                plate_image = resized_image[ymin:ymax, xmin:xmax].copy()
                ocr_text = self.ocr(plate_image)
                # 將結果存下來用於繪製結果圖
                detections_results.append((label, confidence, (xmin, ymin, xmax, ymax), ocr_text))
        # 利用偵測結果繪製結果圖
        self.bounding_image = self.draw_bounding(detections_results, resized_image)
        # 轉型成不同格式的圖片
        self.pillow_image = Image.fromarray(self.bounding_image)
        self.result_image = ImageTk.PhotoImage(self.pillow_image.resize((400, 400)))
        # 將結果圖片顯示在右方畫布
        self.result_image_canvas.create_image(0, 0, anchor = "nw", image = self.result_image)
    
    # 文字辨識
    def ocr(self, image):
        # 透過線性調整加強圖片的特徵
        adjusted = cv2.convertScaleAbs(image, alpha = 1.2, beta = 1.0)
        # 把 RGB 轉為灰階圖
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

        # 透過 threshold 二分法讓圖片變為只有黑白兩種顏色
        ret, adjusted = cv2.threshold(adjusted, 150, 255, cv2.THRESH_BINARY)
        
        # 餵入 OCR 文字辨識      
        ocr_text = pytesseract.image_to_string(adjusted, lang='eng', config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # 設定 regex 代表白名單的文字
        hant_rule = re.compile(r'[A-Z0-9]+')

        # 將符合白名單的文字接在一起即為結果
        result = ''.join(hant_rule.findall(ocr_text))
        return result
    
    # 繪畫偵測邊框
    def draw_bounding(self, detections, image):
        for label, confidence, points, ocr_text in detections:
            left, top, right, bottom = points
            label = ocr_text
            # 透過四個點的座標來繪製綠色邊框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            # 在邊框上方寫上對應的偵測車牌號碼、該位置是車牌的可信度
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image

# 建立 APP
root = tk.Tk()
app = Application(root)

root.mainloop()