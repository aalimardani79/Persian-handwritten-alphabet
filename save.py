
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Path to your 'Persian-Character-DB-Training.cdb' file
filename = 'C:/Users/asus/Desktop/Files/dars/bi/HCD/HCD/Persian-Character-DB-Test.cdb'

MAX_COMMENT = 512

save_images = True
folder_path = 'C:/Users/asus/Desktop/Files/dars/bi/HCD/HCD/u'  # Folder path to save the images

with open(filename, 'rb') as fid:
    # read private header
    header = np.fromfile(fid, dtype=np.uint8, count=7)
    yy = np.fromfile(fid, dtype=np.uint16, count=1)[0]
    m = np.fromfile(fid, dtype=np.uint8, count=1)[0]
    d = np.fromfile(fid, dtype=np.uint8, count=1)[0]
    W = np.fromfile(fid, dtype=np.uint16, count=1)[0]
    H = np.fromfile(fid, dtype=np.uint16, count=1)[0]
    TotalRec = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    nMaxCount = np.fromfile(fid, dtype=np.uint16, count=1)[0]
    LetterCount = np.fromfile(fid, dtype=np.uint32, count=nMaxCount)
    imgType = np.fromfile(fid, dtype=np.uint8, count=1)[0]  # 0: binary, 1: gray
    comments = fid.read(MAX_COMMENT).decode('utf-8')
    print(comments)
    reserved = np.fromfile(fid, dtype=np.uint8, count=490)

    normal = (W > 0) and (H > 0)

    for i in range(TotalRec):
        start_word = np.fromfile(fid, dtype=np.uint16, count=1)[0]  # Must be 0xFFFF
        label = np.fromfile(fid, dtype=np.uint16, count=1)[0]  # Correct label of the character
        confidence = np.fromfile(fid, dtype=np.uint16, count=1)[0]  # Not important

        if not normal:
            W = np.fromfile(fid, dtype=np.uint16, count=1)[0]
            H = np.fromfile(fid, dtype=np.uint16, count=1)[0]

        byte_count = np.fromfile(fid, dtype=np.uint16, count=1)[0]
        image_data = np.zeros((H, W), dtype=np.uint8)

        if imgType == 0:  # Binary
            for y in range(H):
                b_white = True
                counter = 0
                while counter < W:
                    wb_count = np.fromfile(fid, dtype=np.uint8, count=1)[0]
                    image_data[y, counter:counter + wb_count] = 255 if b_white else 0
                    b_white = not b_white
                    counter += wb_count
        else:  # Grayscale mode
            image_data = np.fromfile(fid, dtype=np.uint8, count=W * H).reshape((H, W)).T

        if save_images:
            image_path = os.path.join(folder_path, f'{i}_{label}.png')
            cv2.imwrite(image_path, image_data)

