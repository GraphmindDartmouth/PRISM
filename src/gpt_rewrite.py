import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import time
import argparse


def gpt_response(description, openai_api_key, max_attempts=10, retry_delay=5):
    prompt = f'''

    My data has one background color tag (### Background color style) and two objects with their location ([Left], [Right], [Top] and [Bottom]) and descriptions.
    Given the data, rewrite it to make sure every word in the sentence gets a PropBank-style annotation, and then assign PropBank-style annotation to each word. Here are the requirements for each sentence:
    Here are the requirements for each sentence: 
    (1). Make sure every sentence only has one predicate and clearly focuses on the predicate, you can use adverbs to replace other verbs.
    (2). Make sure the object is assigned ARG0 or ARG1. 
    (3). Make sure the assigned PropBank-style annotation is selected from the following list [ARG0, ARG1, ARG2, ARGM-LOC (Locative), ARGM-MNR (Manner), ARGM-ADV (Adverbial), ARGM-DIR (Direction), ARGM-PRP (Purpose)]
    (4). Avoid passive voice and copular + present participle construction. Use Explicit Active Structure. 
    (5). Make sure the rewritten sentence is simple and you can remove unnecessary words but do not add things that never appear in the original sentence. 
    (6). Add the background color style and position at the end of the description of each object.

    For example:
    [ARG0: The man][V: stands][ARGM-LOC: near the sidewalk edge]. [ARG0: The man][V: stands][ARGM-LOC: close to the building wall] [Background color style: Grey urban] [Position: Left].
    [ARG0: The suitcase][V: is][ARGM-LOC: in the man’s hand]. [ARG0: The suitcase][V: is][ARGM-LOC: beside the man][ARGM-MNR: carried in the man’s hand] [Background color style: Grey urban] [Position: Right].

    Now, given the data "{description}", rewrite the description with EXACTLY the example format:
    '''

    model = ChatOpenAI(model="gpt-4o-mini",
                       openai_api_key=openai_api_key,
                       temperature=0,
                       max_tokens=None,
                       timeout=None,
                       max_retries=2)

    messages = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
        ],
    )

    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        try:
            response = model.invoke([messages])
            if response and response.content and len(response.content.strip()) > 0:
                return response.content
            else:
                print(f"Empty response received on attempt {attempts}. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempts}: {str(e)}")

        if attempts < max_attempts:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)

    raise Exception(f"Failed to get response after {max_attempts} attempts")



def process_item(description, openai_api_key):
    response = gpt_response(description, openai_api_key)
    return response


def split_rewrite_response(sentences):
    """Split multi-line GPT responses into per-object, per-sentence lists."""
    structured = []
    for resp in sentences:
        lines = [line.strip() for line in resp.strip().splitlines() if line.strip()]
        per_image = []
        for line in lines:
            segments = [segment.strip() for segment in line.split('.') if segment.strip()]
            for segment in segments:
                if not segment.endswith('.'):
                    segment = f"{segment}."
                per_image.append(segment)
        structured.append(per_image)
    return structured


import concurrent.futures
def get_data_responses(description_list, openai_api_key):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_item, description_list[i], openai_api_key) for i in range(len(description_list))]
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
openai_api_key = args.openai_api_key
split = "train"
response_list = np.load(f'{save_dir}/brain_data_{split}_final_cc.npz', allow_pickle=True)
res = response_list['response_list']
rewrite_response = get_data_responses(res, openai_api_key)
print(len(rewrite_response))
structured_response = split_rewrite_response(rewrite_response)

np.savez(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz',
         response_list=rewrite_response,
         structured_response=structured_response)

split = "val"
response_list = np.load(f'{save_dir}/brain_data_{split}_final_cc.npz', allow_pickle=True)
res = response_list['response_list']
rewrite_response = get_data_responses(res, openai_api_key)
print(len(rewrite_response))
structured_response = split_rewrite_response(rewrite_response)
np.savez(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz',
         response_list=rewrite_response,
         structured_response=structured_response)

split = "test"
response_list = np.load(f'{save_dir}/brain_data_{split}_final_cc.npz', allow_pickle=True)
res = response_list['response_list']
rewrite_response = get_data_responses(res, openai_api_key)
print(len(rewrite_response))
structured_response = split_rewrite_response(rewrite_response)
np.savez(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz',
         response_list=rewrite_response,
         structured_response=structured_response)
