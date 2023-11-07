import pandas as pd
import numpy as np
import math, cv2
import os
from tqdm import tqdm
import random
import time

def future_trend(df, idx, n_days):
    cur_val = df.Close[idx]
    future_val = df.Close[idx + n_days]
    if cur_val < future_val:
        return "_up"
    elif cur_val == future_val:
        return "_same"
    else:
        return "_down"

def flag_old_years(date_in_some_format):
    date_as_string = str(date_in_some_format)  # cast to string
    year_as_string = int(date_in_some_format[:4]) # last four characters
    if year_as_string < 1993:
        return "too_old"
    else:
        return date_as_string

def write_n_day_images(df, n_days = 20, img_height = 64, path = "None", acronym = "text"):

    df.index = pd.to_datetime(df.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    df.index = df.index.map(flag_old_years)
    df = df[df.index != "too_old"]
    df.index = range(len(df))
    num_images = len(df)
    i = 0
    open_ = [] 
    high_ = []
    low_ = []
    close_ = []
    volume_ = []
    moving_average = []
    past_n_closes = []

    figure_names = []
    images = []
    labels = []

    # generate images
    for index, row in tqdm(df.iterrows()):
        # SMA = average closing price
        if index < n_days:
            past_n_closes.append(row.Close)
        open_.append(row.Open)
        high_.append(row.High)
        low_.append(row.Low)
        close_.append(row.Close)
        volume_.append(row.Volume)           
        i += 1
        if i % n_days == 0:
            if index + n_days < num_images:
                trend = future_trend(df, index, n_days)
            else:
                break
            i = 0

            stock_image = generate_img_from_list(open_, high_, low_, close_, volume_, img_height, past_n_closes)

            if stock_image.size > 0:
                stock_image = np.transpose(stock_image, (2, 1, 0))
                img = cv2.merge((stock_image[2], stock_image[1], stock_image[0]))
                fig_name = str(n_days) + "_days_" + acronym + "_num_" + str(index - n_days + 1) + trend + ".png"
                figure_names.append(fig_name)
                images.append(img)
                labels.append(trend)
                past_n_closes = close_
            else:
                past_n_closes = None

            open_ = [] 
            high_ = []
            low_ = []
            close_ = []
            volume_ = []

    if not images:
        return
    
    # write images
    c = list(zip(figure_names, images, labels))
    random.shuffle(c)
    figure_names, images, labels = zip(*c)
    num_upwards = labels.count("up")
    num_downwards = len(labels) - num_upwards
    labels = [0 if x == "up" else (1 if x == "down" else -1) for x in labels]
    if num_upwards < num_downwards:
        # delete downward
        num_diff = num_downwards - num_upwards
        indices = [i for i, x in enumerate(labels) if x == 0]
        indices = indices[:num_diff]
        figure_names = [item for i, item in enumerate(figure_names) if i not in indices]
        images = [item for i, item in enumerate(images) if i not in indices]
    elif num_upwards > num_downwards:
        # delete upward
        num_diff = num_upwards - num_downwards
        indices = [i for i, x in enumerate(labels) if x == 1]
        indices = indices[:num_diff]
        figure_names = [item for i, item in enumerate(figure_names) if i not in indices]
        images = [item for i, item in enumerate(images) if i not in indices]
    else:
        pass

    for img, fig_name in zip(images, figure_names):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fig_name = os.path.join(path, fig_name)
        cv2.imwrite(fig_name, img)


def generate_img_from_list(open_, high_, low_, close_, volume_, img_height, past_n_closes):
    n_days = len(open_)
    moving_averages = []
    if past_n_closes:
        for i in range(len(close_)):
            moving_averages.append(sum(past_n_closes[i:] + close_[:i]) / n_days)
    else:
        for i in range(len(close_)):
            moving_averages.append(-1)
        
    vol_height = round(img_height * (5/4)) - img_height
    if past_n_closes:
        max_price = max(high_ + moving_averages + open_ + close_ + past_n_closes)
        min_price = min(low_ + moving_averages + open_ + close_ + past_n_closes)
    else:
        max_price = max(high_ + open_ + close_)
        min_price = min(low_ + open_ + close_)

    max_volume = max(volume_)
    min_volume = min(volume_)
    price_range = max_price - min_price
    volume_range = max_volume - min_volume
    black_image_price = np.zeros(img_height*n_days*3*3)
    black_image_price = black_image_price.reshape((img_height, n_days*3, 3))
    black_image_volume = np.zeros(vol_height*n_days*3*3)
    black_image_volume = black_image_volume.reshape((vol_height, n_days*3, 3))

    idx = 0
    past_moving_averages = []
    moving_average_pixel_list = []

    open_naan = len([0 for x in open_ if math.isnan(x)])
    close_naan = len([0 for x in close_ if math.isnan(x)])
    low_naan = len([0 for x in low_ if math.isnan(x)])
    high_naan = len([0 for x in high_ if math.isnan(x)])
    naan_bread = [open_naan, close_naan, low_naan, high_naan]
    max_amount_of_naan_bread = max(naan_bread)

    for o, h, l, c, v, ma in zip(open_, high_, low_, close_, volume_, moving_averages):

        if price_range <= 0 or volume_range <= 0 or max_amount_of_naan_bread > 1:
            empty_array = []
            return np.asarray(empty_array)
        
        check_naan_bread = [o, h, l, c, v, ma]
        if len([0 for x in check_naan_bread if math.isnan(x)]) > 0:
            continue
        
        if moving_averages[0] > -1:
            moving_average_pixel = round((ma - min_price) / price_range * (img_height-1))
            moving_average_pixel_list.append(moving_average_pixel)

        open_pxl = round((o - min_price) / price_range * (img_height-1))
        close_pxl = round((c - min_price) / price_range * (img_height-1))
        vertical_bar_first_pxl = round((l - min_price) / price_range * (img_height-1))
        vertical_bar_last_pxl = round((h - min_price) / price_range * (img_height-1))
        volume_pxl = round((v - min_volume) / volume_range * (vol_height-1))
                
        black_image_price[open_pxl, 3*idx].fill(255)
        if moving_averages[0] > -1:
            black_image_price[moving_average_pixel, 3*idx+1].fill(255)

        if vertical_bar_last_pxl - vertical_bar_first_pxl > 0:
            for inner_idx in range(vertical_bar_last_pxl - vertical_bar_first_pxl):
                black_image_price[vertical_bar_first_pxl + inner_idx, 3*idx+1].fill(255)
        elif vertical_bar_last_pxl - vertical_bar_first_pxl == 0:
            black_image_price[vertical_bar_last_pxl, 3*idx+1].fill(255)
        else:
            pass

        black_image_price[close_pxl, 3*idx+2].fill(255)

        if volume_pxl > 0:
            for inner_idx_2 in range(volume_pxl):
                black_image_volume[inner_idx_2, 3*idx].fill(255)
        elif volume_pxl == 0:
            black_image_volume[0, 3*idx].fill(255)
        else:
            pass

        idx += 1

    if moving_averages[0] > -1:
        past_pixel = moving_average_pixel_list[0]
        for i in range(len(moving_average_pixel_list)-1):
            current_pixel = moving_average_pixel_list[i]
            future_pixel = moving_average_pixel_list[i+1]
            first_column_pixel = round((current_pixel + past_pixel) / 2)
            last_column_pixel =  round((future_pixel + current_pixel) / 2)
            black_image_price[first_column_pixel, 3*i].fill(255)
            black_image_price[last_column_pixel, 3*i+2].fill(255)
            past_pixel = current_pixel
    full_stock_image = np.concatenate((black_image_volume, black_image_price), axis = 0)

    return full_stock_image