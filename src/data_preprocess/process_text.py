import os

import numpy as np
import pandas as pd

def extract_ocr(sample_files, ocr_files):
    OCR = pd.read_csv(ocr_files)
    # traverse stages
    for stage in os.listdir(sample_files):
        # traverse all image folder in stage
        stage_path = os.path.join(sample_files, stage)
        for image_folder in os.listdir(stage_path):
            image_folder_path = os.path.join(stage_path, image_folder)
            
            if not os.path.isdir(image_folder_path):
                # print(image_folder_path)
                continue
            # print(image_folder_path)
            images = os.listdir(image_folder_path)
            panel_id = images[1].split("_")[0]
            comic = OCR[OCR["comic_no"] == int(image_folder)]
            page = comic[comic["page_no"] == int(panel_id)]
            with open(os.path.join(image_folder_path, "text.txt"), "w") as f:
                for i in range(4):
                    panel = page[page["panel_no"] == i]
                    texts = panel["text"].to_list()
                    texts = [str(text) for text in texts if text is not np.nan]
                    if len(texts) == 0:
                        all_text = ""
                    else:
                        all_text = "; ".join(texts)
                    f.write(all_text)
                    f.write("\n")

if __name__ == "__main__":
    ocrs = "/Users/tomoyoshikimura/Documents/fa23/cs546/comic-gen/COMICS_ocr_file.csv"
    samples = "./sample/"
    extract_ocr(samples, ocrs)