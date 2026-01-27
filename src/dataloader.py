import pandas as pd

def load_captions_data(captions_file):
    print(f"Đang đọc file caption từ: {captions_file}")

    # Đọc file CSV (Flickr8k thường có header là 'image' và 'caption')
    df = pd.read_csv(captions_file)

    # Chuyển đổi từ DataFrame sang Dictionary: {tên_ảnh: [list_5_captions]}
    data_dict = {}
    for img_name, caption in zip(df['image'], df['caption']):
        # img_name.strip() để xóa khoảng trắng thừa nếu có
        # str(caption) để đảm bảo caption là string (tránh lỗi NaN)
        img_name = img_name.strip()
        caption = str(caption)

        if img_name not in data_dict:
            data_dict[img_name] = []

        data_dict[img_name].append(caption)

    print(f"Đã load xong dữ liệu của {len(data_dict)} ảnh.")
    return data_dict