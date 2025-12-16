import pandas as pd
import fitz # PyMuPDF
import numpy as np
import os
from tqdm import tqdm
import json

def pdf_data_extraction(path):
    """
    input: BP pdf file
    output: BP vision data and BP text data
    vision data --> ./data/BP-file-np/filename.npz
    text data  --> ./data/BP-text/BP_pdf_texts.json
    """
    pdf_path = path #'./data/BP_PDF/'
    pdf_list = os.listdir(pdf_path)
    pdf_text_json = {}
    error_pdf_list = []
    for pdf in tqdm(pdf_list):
        try:
            pdf_document = fitz.open(pdf_path+'/'+pdf)
            imgs = []# 创建空特征向量
            texts=[] 
            for i in range(pdf_document.page_count):# 遍历每页
                # 获取当前页的图像
                page = pdf_document[i]
                pix = page.get_pixmap()
                # 将图像转换为 NumPy 数组
                img = np.frombuffer(pix.samples, np.uint8).reshape((pix.h, pix.w, pix.n))
                imgs.append(img)

                texts.append(page.get_text(sort=True))
            np.savez_compressed('./data/BP-file-np/'+pdf[:-4]+'.npz',np.array(imgs))
            pdf_text_json[i] = texts
        except:
            error_pdf_list.append(pdf)
    with open('./data/BP-text/BP_pdf_texts.json', 'w',encoding='utf-8') as f:
        json.dump(pdf_text_json,f,ensure_ascii=False)

    return error_pdf_list


def main():
    pdf_data_extraction('./data/BP_PDF/')

if __name__ == "__main__":
    main()