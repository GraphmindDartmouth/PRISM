import numpy as np
import PIL.Image
import io
import base64
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import time
import argparse
import json

from src_final.new_nsd_data import load_data_lists


def convert_pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')  # Saves to memory buffer, not disk
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def gpt_response(img, caption, keyword, openai_api_key, max_attempts=10, retry_delay=5):
    img = convert_pil_to_base64(img)
    if caption:
        prompt = f'''
    Given the image and caption, first describe the background color style of the image with 3-5
    words. Second, detect the TWO most important objects in the image. Then, describe each of the
    objects and their relationship using: {keyword} with TWO sentences. For each sentence, use 5-
    10 words and as easy as possible.

    Then, detect the absolute position of the two objects in the image, and select from [right, left, top,
    bottom]. "left" and "right" should appear together for horizontal objects, and "top" and "bottom"
    should appear together for vertical objects. DO NOT mix.
    Example:

    ### Background color style: Grayscale urban.

    ### The Man [left]
    1. The man is standing near the sidewalk edge. The Man is close to the building wall.

    ### The Suitcase [right]
    1. The suitcase is beside the man's foot. The Suitcase is placed on the street's curved edge.


    Now, given the image I uploaded and the caption "{caption}", detect the two most important
    objects with absolute position, describe them using {keyword} with EXACTLY the example
    format:
    '''
    else:
        prompt = f'''
    Given the image, first describe the background color style of the image with 3-5
    words. Second, detect the TWO most important objects in the image. Then, describe each of the
    objects and their relationship using: {keyword} with TWO sentences. For each sentence, use 5-
    10 words and as easy as possible.

    Then, detect the absolute position of the two objects in the image, and select from [right, left, top,
    bottom]. "left" and "right" should appear together for horizontal objects, and "top" and "bottom"
    should appear together for vertical objects. DO NOT mix.
    Example:
    ### Background color style: Grayscale urban.

    ### The Man [left]
    1. The man is standing near the sidewalk edge. The Man is close to the building wall.

    ### The Suitcase [right]
    1. The suitcase is beside the man's foot. The Suitcase is placed on the street's curved edge.

    Now, given the image I uploaded, detect the two most important
    objects with absolute position, describe them using {keyword} with EXACTLY the example
    format:
    '''


    model = ChatOpenAI(model="gpt-4o-mini",
                       openai_api_key=openai_api_key,
                       temperature=0,
                       max_tokens=None,
                       timeout=None,
                       max_retries=2)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            },
        ],
    )

    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        try:
            response = model.invoke([message])
            if response and response.content and len(response.content.strip()) > 0:
                return response.content
            else:
                print(f"Empty response received on attempt {attempts}. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempts}: {str(e)}")

        # Only sleep if we're going to retry
        if attempts < max_attempts:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)

    # If we've exhausted all attempts, raise an exception
    raise Exception(f"Failed to get response after {max_attempts} attempts")


def process_item(cur_caption, cur_img, keyword, openai_api_key):
    response = gpt_response(cur_img, cur_caption, keyword, openai_api_key)
    return response


import concurrent.futures
def get_data_responses(caption_list, img_list, keyword, openai_api_key):
    if len(caption_list) == 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, None, img_list[i], keyword, openai_api_key) for i in range(len(img_list))]
            response_list = [future.result() for future in futures]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, caption_list[i], img_list[i], keyword, openai_api_key) for i in range(len(img_list))]
            response_list = [future.result() for future in futures]
    return response_list




parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_split",
    type=str,
    default="train",
    choices=['train', 'val', 'test'],
    required=True,
    help="data split"
)

parser.add_argument(
    "--best_keyword_file",
    type=str,
    default="best_keyword.json",
    help="Path to the best keyword JSON file")

parser.add_argument(
    "--openai_api_key",
    type=str,
    required=True)

parser.add_argument(
    "--save_dir",
    type=str,
    default="data"
)





args = parser.parse_args()
data_split = args.data_split
save_dir = args.save_dir
keyword_save_path = args.best_keyword_file
openai_api_key = args.openai_api_key

data = load_data_lists(data_split, save_dir)
fmri_train = data[0]
img_train = data[1]
caption_train = data[2]
coco_id_train = data[3]
print("Data loaded! Length: ", len(fmri_train), len(img_train), len(caption_train), len(coco_id_train))
assert len(fmri_train) == len(img_train) == len(coco_id_train)

with open(keyword_save_path, "r") as f:
    data = json.load(f)

# Since the file contains only one key-value pair
best_keyword = next(iter(data))
best_score = data[best_keyword]

print("Best keyword:", best_keyword)
print("Score:", best_score)


# train
response_list = get_data_responses(caption_train, img_train, best_keyword, openai_api_key)
print(len(response_list))
# save
np.savez(f'{save_dir}/{data_split}_description.npz', response_list=response_list)


