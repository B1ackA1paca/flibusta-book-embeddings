import os
import json
import re
import zipfile
import pandas as pd
import numpy as np

def get_data_df(inpx_path="/data/flibusta/data/fb2.flibusta.lib.rus.ec.7z.inpx"):
    summarize_matrix = []   
    with zipfile.ZipFile(inpx_path) as z:
        for name in z.namelist():
            if name.endswith('.inp'):
                archive_name = os.path.basename(name).replace('.inp', '')
                with z.open(name) as f:
                    text = f.read().decode('utf-8', errors='ignore')
                    text_list = text.split("\x04")
                    text_matrix = [text_list[i:i+17] + [archive_name] for i in range(0, len(text_list), 17)]
                    summarize_matrix += text_matrix

    summarize_df = pd.DataFrame(summarize_matrix)
    df_columns = ["author", "genre", "name", "series", "number_in_series", "file_number", "size_in_bites", "flibusta_id", "flag_del", "format", "date", "folder_number", "lang", "rating", "unknown", "year", "site", "archive"]
    summarize_df.columns = df_columns

    trash = r'[\r\n]'
    summarize_df['author'] = summarize_df['author'].astype(str).str.replace(trash, '', regex=True)

    data_df = summarize_df[(summarize_df.isna().sum(axis=1) <= 10)].reset_index(drop=True)
    data_df = data_df[(data_df["format"] == "fb2") & (data_df["site"] == "Flibusta")].reset_index(drop=True)
    data_df = data_df[("/data/flibusta/data/" + data_df['archive'] + '/' + data_df['file_number'] + ".fb2").map(os.path.exists)].reset_index(drop=True)

    return data_df

def get_checkpoints_df(inpx_path="/data/flibusta/data/fb2.flibusta.lib.rus.ec.7z.inpx", folder="checkpoints"):
    data_df = get_data_df()

    data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data.append(json.load(f))

    checkpoints_df = pd.DataFrame(data).rename(columns={"book_id": "file_number"}).merge(data_df[["file_number", "archive"]], on="file_number")
    return checkpoints_df

def get_genres(data_df, genre_column="genre"):
    arr = data_df[genre_column].str.split(":").values
    ans = []
    for lst in arr:
        ans += lst
    return np.unique(np.array(ans))[1:]

def read_fb2(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        
    annot = re.findall(r'<annotation>(.*?)</annotation>', text, re.S)
    sections = re.findall(r'<section>(.*?)</section>', text, re.S)
    
    return [re.sub(r'</?[a-zA-Z][^>]*>', '', chunk).strip() for chunk in (annot + sections) if chunk]

def clean_text(text):
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[#@]\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()