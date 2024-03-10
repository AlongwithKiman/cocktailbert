import torch
from cocktailbert import BERTClassification
from kobert_tokenizer import KoBERTTokenizer
from utils import postprocess_model
import json
import argparse
from argparse import ArgumentParser
import pandas as pd
from utils import ColorComparator
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


def show_images(df, font_path):
    width, height = 800, 200
    margin = 10
    font_size = 14
    images_per_row = 4
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new('RGB', (width, int(len(df) * height / images_per_row + 100)), color='white')
    draw = ImageDraw.Draw(img)
    image_width = (width - margin * (images_per_row + 1)) // images_per_row
    image_height = height - 2 * margin - font_size

    for i, (_, row) in enumerate(df.iterrows()):
        response = requests.get(row['image'])
        img_temp = Image.open(BytesIO(response.content))
        img_temp = img_temp.resize((image_width, image_height), Image.ANTIALIAS)
        img.paste(img_temp, ((i % images_per_row) * (image_width + margin) + margin, (i // images_per_row) * height + margin))

        text = f"{i+1}. {row['name']} (ABV: {row['ABV']}%)"
        text_width, text_height = draw.textsize(text, font=font)
        draw.text(((i % images_per_row) * (image_width + margin) + margin + (image_width - text_width) // 2, (i // images_per_row) * height + image_height + margin), text, font=font, fill='black')
    img.show()

if __name__ == "__main__":
    
    # argparse
    parser = ArgumentParser()
    parser.add_argument('--config', default="./config/inference.json", type=str)
    parser.add_argument('--sentence', default="데이트 할 때 마시기 좋은 달달한 칵테일", type=str)
    parser.add_argument('--visualize', default=False, action="store_true")
    parser.add_argument('--font_path', default=None, type=str)
    args = parser.parse_args()

    if args.sentence == None:
        raise ValueError("sentence arguement에 칵테일의 느낌을 입력해주세요")

    if args.visualize and not args.font_path:
        raise ValueError("To visualize cocktails, set font path")

    args = parser.parse_args()
    with open(args.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)   

    # load model
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    loaded_ckpt = torch.load(hparam.checkpoint_path)
    loaded_model={}
    for key, value in loaded_ckpt.items():
        loaded_model[key] = value

    new_model = BERTClassification([hparam.num_size_category, hparam.num_ABV_category, hparam.num_color_category])
    new_model.load_state_dict(loaded_model, strict=False)
    new_model.eval()


    # run model & postprocess
    size, abv, color = postprocess_model(new_model,args.sentence, tokenizer)
    abv_min, abv_max = abv

    if color == "red":
        color = "ff0000"
    elif color == "green":
        color = "00ff00"
    elif color == "blue":
        color = "33ffff"
    elif color == "brown":
        color = "da8c17"
    elif color == "yellow":
        color = "ffff00"
    else:
        raise ValueError("invalid abv type")

    #recommend cocktails with model result
    df = pd.read_csv(hparam.cocktail_list_path)
    
    # filter by size, ABV
    filtered_df = df[(df['filter_type_two'] == size) & (df['ABV'] >= abv_min) & (df['ABV'] <= abv_max)]

    # sort by color similarity
    color_comparator = ColorComparator()
    filtered_df['similarity'] = filtered_df['color'].apply(lambda x: color_comparator.color_similarity(color1=x, color2=color))
    filtered_df_sorted = filtered_df.sort_values(by='similarity', ascending=True)    
    filtered_df_sorted.drop(columns=['similarity'], inplace=True)
    filtered_df_sorted.reset_index(drop=True, inplace=True)
    

    print(f"########## {color}, abv range in {abv}, {size} drinks below ##########")
    if args.visualize:
        show_images(filtered_df_sorted,args.font_path)
    else:
        print(filtered_df_sorted)



