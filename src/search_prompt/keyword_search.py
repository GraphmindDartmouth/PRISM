import sys
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, StableDiffusionPipeline
from sklearn.metrics.pairwise import rbf_kernel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from torchvision import transforms
import lpips

import io
import base64
import time

from keyword_generator import keyword_generator
import json
import os
from src.new_nsd_data import load_data_lists
import argparse


device = "cuda:0"
print(f"Using device: {device}")
model_id = "stabilityai/stable-diffusion-2-1"

vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
vae = vae.to(device)

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

text_encoder = text_encoder.to(device)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to(device)

lpips_model = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'


def centering(K):
    """Center the kernel matrix K"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def rbf_CKA(X, Y, sigma=None):
    """Compute RBF kernel CKA between feature matrices X and Y"""
    # Calculate RBF kernel matrices with auto sigma if not provided
    K = rbf_kernel(X, gamma=sigma)
    L = rbf_kernel(Y, gamma=sigma)

    # Center the kernel matrices
    K_centered = centering(K)
    L_centered = centering(L)

    # Compute HSIC
    HSIC = np.sum(K_centered * L_centered)

    # Normalize
    HSIC_XX = np.sum(K_centered * K_centered)
    HSIC_YY = np.sum(L_centered * L_centered)

    # Return CKA
    return HSIC / np.sqrt(HSIC_XX * HSIC_YY)



def get_text_embeddings(text, openai_api_key, device=None):
    if device is None:
        device = device

    text_input = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt")

    text_input = {k: v.to(device) for k, v in text_input.items()}
    with torch.no_grad():
        text_embeddings = text_encoder(**text_input)[0]

    return text_embeddings


def convert_pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')  # Saves to memory buffer, not disk
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def gpt_response(img, caption, keyword, openai_api_key ,max_attempts=10, retry_delay=5):
    img = convert_pil_to_base64(img)

    prompt = f'''
    Given the image and caption {caption}, describe the image I upload using the keyword: {keyword}.
    '''

    model = ChatOpenAI(model="gpt-4o-mini",
                       openai_api_key=openai_api_key,
                       temperature=0.2,
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

        if attempts < max_attempts:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)

    raise Exception(f"Failed to get response after {max_attempts} attempts")




def process_item(cur_caption, cur_img, keyword, openai_api_key):
    response = gpt_response(cur_img, cur_caption, keyword, openai_api_key)
    return response

import concurrent.futures
def get_data_responses(caption_list, img_list, keyword, openai_api_key):
    assert len(caption_list) == len(img_list)
    response_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_item, caption_list[i], img_list[i], keyword, openai_api_key) for i in range(len(caption_list))]
        response_list = [future.result()[0] for future in futures]
    return response_list



def cal_lpip_pix(obj_des_list, gt_img_list):
    # cal lpip
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    image_tensors = []
    gt_img_tensors = []
    for i in range(len(obj_des_list)):
        obj, content = obj_des_list[i]
        objs = obj[0] + " " + obj[1]
        prompt = [objs, content[0], content[1]]
        generated_img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
        generated_img = transform(generated_img[0])
        image_tensors.append(generated_img)
        gt_img = transform(gt_img_list[i])
        gt_img_tensors.append(gt_img)
    image_batch = torch.stack(image_tensors)
    gt_img_batch = torch.stack(gt_img_tensors)

    batch_size = image_batch.shape[0]
    lpips_scores = []

    for i in range(batch_size):
        score = lpips_model(image_batch[i].unsqueeze(0), gt_img_batch[i].unsqueeze(0))
        lpips_scores.append(score.item())

    lpips_score = sum(lpips_scores) / batch_size

    # pixcorr
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    all_images_flattened = preprocess(image_batch).reshape(len(image_batch), -1).cpu()
    all_images_gt_flattened = preprocess(gt_img_batch).view(len(gt_img_batch), -1).cpu()

    print(all_images_flattened.shape)
    print(all_images_gt_flattened.shape)

    corrsum = 0
    for i in range(len(all_images_flattened)):
        corrsum += np.corrcoef(all_images_flattened[i], all_images_gt_flattened[i])[0][1]
    corrmean = corrsum / len(image_batch)

    pixcorr = corrmean
    return lpips_score, pixcorr




def cal_score(fmri, captions, imgs, keyword, openai_api_key):
    print("Calculating score for keyword:", keyword)
    response_list = get_data_responses(captions, imgs, keyword, openai_api_key)
    for i in range(0, len(response_list)):
        response_file = f"response/response{i}.json"
        os.makedirs(os.path.dirname(response_file), exist_ok=True)
        if os.path.exists(response_file):
            with open(response_file, "r") as f:
                response_dict = json.load(f)
        else:
            response_dict = {}
        response_dict[keyword] = response_list[i]
        with open(response_file, "w") as f:
            json.dump(response_dict, f, indent=4)

    lpips_score, pixcorr = cal_lpip_pix(response_list, imgs)

    lpip_score = 1 - lpips_score
    text_emb_list = []
    for i in response_list:
        text_emb = get_text_embeddings(i, device)
        text_emb_list.append(text_emb)
    text_emb_list = torch.stack(text_emb_list)
    text_emb_list = text_emb_list.reshape(text_emb_list.size(0), -1)
    text_emb_list = text_emb_list.cpu().numpy()
    cka = rbf_CKA(fmri, text_emb_list)

    return cka, lpip_score, pixcorr




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--best_keyword_file", type=str, default="best_keyword.json",
                        help="Path to the best keyword JSON file")
    parser.add_argument("--ref_num", type=int)
    parser.add_argument("--exp_num", type=int)
    parser.add_argument("--temp", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--init_relation_json_path", type=str)


    args = parser.parse_args()
    openai_api = args.openai_api_key
    keyword_save_path = args.best_keyword_file

    train_fmri, train_img_list, train_caption_list, _ = load_data_lists(split="train", save_dir=args.save_dir)
    assert len(train_fmri) == len(train_img_list) == len(train_caption_list)
    
    imgs = train_img_list
    captions = train_caption_list
    fmris = train_fmri

    ### Define the reference keyword nums and expanding nums
    REF_NUM = args.ref_num
    EXP_NUM = args.exp_num

    with open(args.init_relation_json_path, "r") as f:
        data = json.load(f)


    initial_relation_names = data

    keywords_dict = {}
    for keyword in initial_relation_names:
        keywords_dict[keyword] = cal_score(fmris, captions, imgs, keyword, openai_api)
        print(f"{keyword}: {keywords_dict[keyword]}")


    generator = keyword_generator(openai_api)
    for i in range(args.epoch):
        # sample some relation names
        # Sort keywords by score
        sorted_keywords = sorted(keywords_dict.keys(), key=lambda x: keywords_dict[x], reverse=True)
        
        temperature = args.temp
        scores = np.array([keywords_dict[k] for k in sorted_keywords])
        scores = np.exp(scores / temperature)  # Apply softmax with temperature
        probabilities = scores / np.sum(scores)
        chosen_indices = np.random.choice(
            len(sorted_keywords), 
            size=min(REF_NUM, len(sorted_keywords)), 
            replace=False, 
            p=probabilities)

        chosen_relation_names = [sorted_keywords[i] for i in chosen_indices]

        print(f"Attempt {i+1} to generate keywords from {chosen_relation_names}.")
        new_keywords = generator.generate_key_word(chosen_relation_names, gen_num=EXP_NUM)
        for keyword in new_keywords:
            if keyword not in keywords_dict:
                scores = cal_score(fmris, captions, imgs, keyword, openai_api)
                keywords_dict[keyword] = scores[0]

        print("Keywords at attempt {i+1}:")
        for keyword, score in keywords_dict.items():
            print(f"{keyword}: {score}")

        ### Save the keywords to json file
        with open("keywords.json", "w") as f:
            json.dump(keywords_dict, f, indent=4)


        sorted_keywords = sorted(keywords_dict.keys(), key=lambda x: keywords_dict[x], reverse=True)
        ### Save only the best keywords to a separate json file
        best_keyword = sorted_keywords[0]
        with open(keyword_save_path, "w") as f:
            json.dump({best_keyword: keywords_dict[best_keyword]}, f, indent=4)



