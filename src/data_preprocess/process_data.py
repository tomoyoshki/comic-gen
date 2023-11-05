import os

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer


def extract_ocr(sample_files, ocr_files):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    OCR = pd.read_csv(ocr_files)
    # traverse stages
    for stage in os.listdir(sample_files):
        # traverse all image folder in stage
        stage_path = os.path.join(sample_files, stage)
        for image_folder in os.listdir(stage_path):
            image_folder_path = os.path.join(stage_path, image_folder)
            if not os.path.isdir(image_folder_path):
                print(f"Skipped: {image_folder_path}")
                continue

            images = os.listdir(image_folder_path)
            panel_id = images[0].split("_")[0]
            if panel_id in {"images.pt", "text.txt"}:
                panel_id = images[1].split("_")[0]
                if panel_id in {"images.pt", "text.txt"}:
                    panel_id = images[2].split("_")[0]

            comic = OCR[OCR["comic_no"] == int(image_folder)]
            page = comic[comic["page_no"] == int(panel_id)]
            all_texts = []
            all_images = []
            for i in range(4):
                # process_images
                image_path = os.path.join(image_folder_path, f"{panel_id}_{i}.pt")
                image_tensor = torch.load(image_path)
                all_images.append(image_tensor)
                # process_text
                panel = page[page["panel_no"] == i]
                texts = panel["text"].to_list()
                texts = [str(text) for text in texts if text is not np.nan]
                if len(texts) == 0:
                    all_text = "None"
                else:
                    all_text = "; ".join(texts)
                all_texts.append(all_text)
            with open(os.path.join(image_folder_path, "text.txt"), "w", encoding="utf-8") as f:
                print(image_folder_path)
                for i in range(4):
                    panel = page[page["panel_no"] == i]
                    texts = panel["text"].to_list()
                    texts = [str(text) for text in texts if text is not np.nan]
                    if len(texts) == 0:
                        all_text = ";"
                    else:
                        all_text = "; ".join(texts)
                    f.write(all_text)
                    f.write("\n")

            all_images = torch.stack(all_images)
            tokens = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]                    
            to_save = torch.stack((input_ids, attention_mask), dim=1)

            torch.save(all_images, os.path.join(image_folder_path, "images.pt"))
            torch.save(to_save, os.path.join(image_folder_path, "text_tokens.pt"))

if __name__ == "__main__":
    ocrs = "/Users/tomoyoshikimura/Documents/fa23/cs546/comic-gen/data/COMICS_ocr_file 17.47.10.csv"
    samples = "/Users/tomoyoshikimura/Documents/fa23/cs546/comic-gen/data/sample"
    extract_ocr(samples, ocrs)