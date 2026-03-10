import pandas as pd
import kagglehub
import os
import shutil



class Flickr8k:
    def __init__(self, path):
        self.path = path
        self.captions_path = os.path.join(path, "captions.txt")
        self.download_data()


    def download_data(self):
        if os.path.exists(self.path):
            print("Data đã tồn tại")
            return
        try:
            cache_path = kagglehub.dataset_download("adityajn105/flickr8k")
            if not os.path.exists(self.path):
                shutil.move(cache_path, self.path)
                print("Tải dữ liệu thành coông")
        except Exception as e:
            print(f"[ERROR] {e}")


    def load_caption(self):
        data_dict = {}
        df = pd.read_csv(self.captions_path)
        for img_name, caption in zip(df['image'], df['caption']):
            img_name = img_name.strip()
            caption = str(caption)


            if img_name not in data_dict:
                data_dict[img_name] = []
            data_dict[img_name].append(caption)
        return data_dict