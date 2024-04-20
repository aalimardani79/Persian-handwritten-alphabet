# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:26:28 2024

@author: asus
"""

import os
import shutil
folder_path = "C:/Users/asus/Desktop/Files/dars/bi/HCD/HCD/train"

def classify_images(folder_path):
    # بررسی وجود فولدر خروجی
    output_folder = os.path.join(folder_path, 'Classified_Images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # خواندن عکس‌ها
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # جدا کردن نام فایل و پسوند
        file_name, extension = os.path.splitext(image_file)
        
        # جدا کردن طبقه‌بندی
        label = file_name.split('_')[-1]

        # ایجاد پوشه
        destination_folder = os.path.join(output_folder, label)

        # بررسی وجود دایرکتوری مقصد
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # کپی عکس به دایرکتوری مقصد
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy(image_path, destination_path)

    print("تقسیم بندی تصاویر با موفقیت انجام شد.")

# فراخوانی تابع با آدرس فولدر عکس‌ها
classify_images(folder_path)
# مثال استفاده از تابع

