#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:14:07 2023

@author: pablofernando
"""


import hashlib
import os
from copy import deepcopy
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from imagehash import average_hash, colorhash, dhash, phash, whash
from PIL import Image
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay



HASH_SIZE = 8

IMG_HASH_FUNCTIONS = {
    'ahash': average_hash,
    'dhash': dhash,
    'phash': phash,
    'whash': whash,
    'chash': colorhash,
}


def download_image(
    url: str,
    save_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    output_fileext: Optional[str] = 'jpg',
) -> Image.Image:
    """
    Downloads an image from the given URL and returns a Pillow image object.

    Args:
        url (str): The URL of the image to download.
        save_dir (str, optional): The directory to save the image to. Defaults to None.
        prefix (str, optional): A prefix to add to the filename when saving the image. Defaults to None.

    Returns:
        Image.Image: A Pillow image object representing the downloaded image.

    Raises:
        ValueError: If the request to download the image fails.
    """

    # Make the request to download the image
    response = requests.get(url)
    # Check that the request was successful
    if response.status_code != 200:
        raise ValueError(f'Failed to download image from URL: {url}')
    # Load the downloaded data into a Pillow image
    image = Image.open(BytesIO(response.content))
    # If a save directory is provided, save the image with the given prefix
    if save_dir is not None:
        # Generate a filename for the image based on the prefix and the image URL
        filename = f"{prefix}_{url.split('/')[-1]}"

        # Save the image to the specified directory
        save_path = os.path.join(save_dir, filename + output_fileext)
        image.save(save_path)
    # Return the Pillow image object
    return image


def download_multiple_images(
    images_url_list: list,
    output_dir: Optional[Union[str, List[str]]] = 'outputs',
    save_to_file: Optional[bool] = True,
    use_cache: Optional[bool] = True,
    image_file_prefix: Optional[str] = 'image',
    show_progress: Optional[bool] = True,
    output_fileext: Optional[str] = 'jpg',
):
    """
    Downloads multiple images from a list of URLs and returns a pandas DataFrame
    containing the downloaded images and their file paths.

    Args:
        images_url_list (list): A list of image URLs to download.
        output_dir (str, optional): The directory to save the images to. Defaults to 'outputs'.
        save_to_file (bool, optional): Whether to save the images to files. Defaults to True.
        use_cache (bool, optional): Whether to use a cache to avoid re-downloading images. Defaults to True.
        image_file_prefix (str, optional): A prefix to add to the filename when saving the images. Defaults to None.
        show_progress (bool, optional): Whether to display a progress bar while downloading the images. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the downloaded images and their file paths.
    """
    if isinstance(output_dir, str):
        output_dir = [output_dir]

    if isinstance(output_dir, list):
        assert (len(output_dir) == 1) or (
            len(output_dir) == len(images_url_list)
        ), 'Invalid output dir list lenght'

    _images_url_list = deepcopy(images_url_list)

    if show_progress:
        _images_url_list = tqdm(_images_url_list)

    downloaded_images = []
    for i, url in enumerate(_images_url_list):

        if len(output_dir) > 1:
            output_dir_ = output_dir[i]
        else:
            output_dir_ = output_dir[0]

        if not os.path.exists(output_dir_):
            os.makedirs(output_dir_)

        # Use the hash of the URL as a cache key
        cache_key = hashlib.md5(url.encode('utf-8')).hexdigest()

        image_filename = f'{image_file_prefix}_{cache_key}.{output_fileext}'
        image_full_path = os.path.join(output_dir_, image_filename)

        if use_cache:

            # Check if the image is already downloaded
            if os.path.exists(image_full_path):

                downloaded_images.append(
                    {
                        'PIL_image_object': Image.open(image_full_path),
                        'file_path': image_full_path,
                        'cache_key': cache_key,
                    }
                )
                print(f'Using cached image for "{url}"')
                continue

        try:
            image = download_image(url)
        except ValueError as e:
            print(f'Failed to download image from URL: {url} - {e} ')
            continue

        if save_to_file:
            # Save the image to the specified directory
            image.save(image_full_path)

            downloaded_images.append(
                {
                    'PIL_image_object': image,
                    'file_path': image_full_path,
                    'cache_key': cache_key,
                }
            )
        else:
            downloaded_images.append(
                {
                    'PIL_image_object': image,
                    'file_path': None,
                    'cache_key': cache_key,
                }
            )

    return pd.DataFrame(downloaded_images)


def hex_to_bin(
    hexstr: str, group_bytes: bool = False
) -> Union[List[bool], np.ndarray]:
    """
    Convert a hexadecimal string to a binary string or a 2D numpy array.

    Args:
        hexstr (str): The hexadecimal string to convert to binary.
        group_bytes (bool, optional): Whether to return a 2D numpy array with shape (sqrt(len(hexstr)*4), sqrt(len(hexstr)*4)) or a list of booleans. Defaults to False.

    Returns:
        Union[List[bool], np.ndarray]: If group_bytes is False, returns a list of booleans representing the binary string. Otherwise, returns a 2D numpy array with shape (sqrt(len(hexstr)*4), sqrt(len(hexstr)*4)).

    Raises:
        ValueError: If hexstr is not a valid hexadecimal string.

    """
    try:
        int(hexstr, 16)
    except ValueError:
        raise ValueError('Invalid hexadecimal string')

    hash_size = int(np.sqrt(len(hexstr) * 4))
    binary_str = format(int(hexstr, 16), f'0{hash_size*hash_size}b')
    binary_list = list(map(lambda x: bool(int(x)), list(binary_str)))
    binary_array = np.array(binary_list).reshape(hash_size, hash_size)

    return binary_array if group_bytes else binary_list


def gen_image_hashes(
    image: Image.Image,
    hash_list: Optional[List[str]] = None,
    hex_only: Optional[bool] = False,
) -> Dict[str, str]:
    """
    Generates image hashes using various hash functions provided by the `imagehash` library.

    Args:
        image (PIL.Image.Image): The image object for which to generate hashes.
        hash_list (list, optional): A list of hash function names to use. Defaults to None.

    Returns:
        dict: A dictionary containing the hash values for the image computed using different hash functions.

    Example:
        # Load an image using Pillow
        image = Image.open('example_image.jpg')

        # Compute image hashes using all hash functions
        hashes = gen_image_hashes(image)

        # Compute image hashes using only a subset of hash functions
        hashes = gen_image_hashes(image, hash_list=['ahash', 'phash'])
    """
    output_hashes = {'hex': {}}

    if not hex_only:
        output_hashes['bool'] = {}

    if not hash_list:
        hash_list = list(IMG_HASH_FUNCTIONS.keys())

    for hash_name in hash_list:
        hash_fn = IMG_HASH_FUNCTIONS[hash_name]
        if hash_name != 'chash':
            hex_hash = str(hash_fn(image, hash_size=HASH_SIZE))
        else:
            hex_hash = str(hash_fn(image, binbits=4)).zfill(16)
        output_hashes['hex'][hash_name] = hex_hash

        if 'bool' in output_hashes:
            output_hashes['bool'][hash_name] = hex_to_bin(hex_hash)

    return output_hashes


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def image_hash_features(
    image: Image.Image,
    hash_fn_list: Optional[List[str]] = IMG_HASH_FUNCTIONS.keys(),
) -> List[bool]:
    hash_bits_array = []
    fn_hashs_concat_order = []

    img_hash = gen_image_hashes(image, hex_only=False)
    bool_hashs = img_hash['bool']

    for hash_fn in hash_fn_list:
        fn_hashs_concat_order.append(hash_fn)

        hash_bits_array.extend(flatten(bool_hashs[hash_fn]))

    return {
        'hash_bits_array': hash_bits_array,
        'fn_hashs_concat_order': fn_hashs_concat_order,
    }


#%%

import json

with open('cat_dog_url_dataset.json', 'r') as f:
    df_samples = pd.DataFrame(json.load(f))

df_samples['output_dir'] = 'outputs/target/' + df_samples.label

df_samples = (
    df_samples[df_samples.url.str.endswith('.jpg')]
    .drop_duplicates()
    .reset_index(drop=True)
)
#%%
images_df = download_multiple_images(df_samples.url, df_samples.output_dir)
images_df['label'] = df_samples.label

label_map = {
    0: 'cat',
    1: 'dog',
    'cat': 0,
    'dog': 1
}

images_df['bin_label'] = [
    label_map.get(label) for label in images_df.label
]
images_df.head()


df_feat = pd.DataFrame(
    images_df.PIL_image_object.apply(image_hash_features).to_list()
)

cols_ = df_feat.columns

images_df[cols_] = df_feat[cols_]


#%%

img_features = pd.DataFrame(np.vstack(images_df['hash_bits_array']))
img_features.columns = [f'img_feature_{c}' for c in img_features.columns]

img_feat_cols_ = img_features.columns.tolist()

images_df.loc[:, img_feat_cols_] = img_features[img_feat_cols_]

#%%

images_df

#%%


X = images_df[img_feat_cols_].copy()
y = images_df.bin_label

#%% Simple PCA test


pca = PCA(n_components=60, svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print('\n', sum(pca.explained_variance_ratio_), '\n')
print(pca.singular_values_)

#%%

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

rfc = RandomForestClassifier(random_state=0, n_jobs=-1, verbose=1)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

#%%


print('accuracy_score: ', accuracy_score(y_test, y_pred))
print('f1_score: ', f1_score(y_test, y_pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=[label_map.get(c) for c in rfc.classes_]
)

disp.plot()

