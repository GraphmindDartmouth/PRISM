import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from gen_pipe import GeneratorPipe
from create_data import load_all_data
from new_nsd_data import load_data_lists
from my_model import DualObjectFMRItoT5
from text_prompt_utils import decoded_sentences_to_prompt_and_direction

import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import open_clip
from transformers import T5TokenizerFast
from torchvision.models import inception_v3, Inception_V3_Weights



def eval_pixcor(all_images, all_recons):
    target_size = (425, 425)
    all_images = F.interpolate(all_images, size=target_size, mode="bilinear", align_corners=False)
    all_recons = F.interpolate(all_recons, size=target_size, mode="bilinear", align_corners=False)


    all_images_np = all_images.detach().cpu().numpy()
    all_recons_np = all_recons.detach().cpu().numpy()

    corrsum = 0
    for i in range(len(all_images_np)):
        img_flat = all_images_np[i].reshape(-1)
        recon_flat = all_recons_np[i].reshape(-1)
        corrsum += np.corrcoef(img_flat, recon_flat)[0][1]
    corrmean = corrsum / len(all_images_np)

    pixcorr = corrmean
    # print(pixcorr)
    return pixcorr




def eval_ssim(all_images, all_recons):
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    if all_images.ndim == 3:
        all_images = all_images.unsqueeze(0)
    if all_recons.ndim == 3:
        all_recons = all_recons.unsqueeze(0)
    target_size = (425, 425)
    all_images = F.interpolate(all_images, size=target_size, mode="bilinear", align_corners=False)
    all_recons = F.interpolate(all_recons, size=target_size, mode="bilinear", align_corners=False)

    img_gray = rgb2gray(all_images.permute(0, 2, 3, 1).cpu().numpy())
    recon_gray = rgb2gray(all_recons.permute(0, 2, 3, 1).cpu().numpy())

    iterator = zip(img_gray, recon_gray)
    ssim_score = []
    for im, rec in iterator:
        ssim_score.append(
            ssim(rec, im, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                 data_range=1.0))

    mean_ssim = np.mean(ssim_score)
    # print(f"SSIM: {mean_ssim:.4f}")
    return mean_ssim


def eval_lpips(all_images, all_recons):
    import lpips
    import torch.nn.functional as F

    if all_images.ndim == 3:
        all_images = all_images.unsqueeze(0)
    if all_recons.ndim == 3:
        all_recons = all_recons.unsqueeze(0)

    # Ensure same resolution
    target_size = (256, 256)
    all_images = F.interpolate(all_images, size=target_size, mode="bilinear", align_corners=False)
    all_recons = F.interpolate(all_recons, size=target_size, mode="bilinear", align_corners=False)

    # Convert [0,1] → [-1,1]
    all_images = all_images * 2 - 1
    all_recons = all_recons * 2 - 1

    loss_fn = lpips.LPIPS(net='alex').to(all_images.device)

    with torch.no_grad():
        lpips_score = loss_fn(all_images, all_recons)

    mean_lpips = lpips_score.mean().item()

    # print(f"LPIPS: {mean_lpips:.4f}")
    return mean_lpips


@torch.no_grad()
def extract_openclip_image_features(images, model, preprocess, device, normalize=True):
    batch = torch.stack([preprocess(img) for img in images], dim=0).to(device)
    feats = model.encode_image(batch)

    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.float().cpu().numpy()


def load_openclip_model(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda:0"):
    """
    Load an OpenCLIP image encoder and its preprocessing transform.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()

    return model, preprocess, device



@torch.no_grad()
def two_way_identification_openclip(all_recons, all_images, model, preprocess, return_avg=True, device="cuda:0"):
    assert len(all_recons) == len(all_images), "Lengths must match."
    n = len(all_images)
    assert n > 1, "Need at least 2 samples."

    preds = extract_openclip_image_features(
        all_recons, model, preprocess, device, normalize=True)
    reals = extract_openclip_image_features(
        all_images, model, preprocess, device, normalize=True)

    r = np.corrcoef(reals, preds)[:n, n:]
    congruents = np.diag(r)  # [N]

    success = r < congruents[None, :]
    success_cnt = success.sum(axis=0)  # [N]

    if return_avg:
        return success_cnt.mean() / (n - 1)
    else:
        return success_cnt, n - 1



@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True, device="cuda:0"):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1




def parse_text_generation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--text_model_ckpt", type=str, default=None,
                        help="Path to the trained DualObjectFMRItoT5 checkpoint.")
    parser.add_argument("--text_batch_size", type=int, default=8,
                        help="Batch size for fmri-to-text generation.")
    parser.add_argument("--text_max_length", type=int, default=100,
                        help="Max decoder length when generating text.")
    parser.add_argument("--text_max_samples", type=int, default=0,
                        help="Limit the number of test samples to process (0 = all).")
    parser.add_argument("--text_output_json", type=str, default=None,
                        help="Optional path to save generated sentences as JSON.")
    parser.add_argument("--text_device", type=str, default=None,
                        help="Device string for fmri-to-text inference, e.g. cuda:0.")
    parser.add_argument("--text_num_tokens", type=int, help="Number of learned tokens used when the text model was trained.")
    parser.add_argument("--text_fmri_dim", type=int, default=15724,
                        help="Dimensionality of each fMRI vector.")
    parser.add_argument("--text_t5_model", type=str, default="t5-base", help="T5 backbone name used during training.")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save data.")
    parser.add_argument("--img_height", type=int)
    parser.add_argument("--img_width", type=int)
    parser.add_argument("--end_steps", type=int)
    parser.add_argument("--base_ratio", type=float)
    parser.add_argument("--num_inference_steps", type=int)




    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def decode_generated_batch(generated_ids, tokenizer):
    """Decode the list of tensors returned by model.generate_sentences."""
    if not generated_ids:
        return []

    batch_size = generated_ids[0].size(0)
    decoded = [[] for _ in range(batch_size)]
    for sent_ids in generated_ids:
        sent_ids = sent_ids.cpu()
        for row_idx in range(batch_size):
            decoded[row_idx].append(
                tokenizer.decode(sent_ids[row_idx].tolist(), skip_special_tokens=True)
            )
    return decoded


def generate_text_from_test_data(test_dataset, gen_args):
    """Generate text for test fMRI samples using a trained DualObjectFMRItoT5 model."""
    if not gen_args.text_model_ckpt:
        return []

    device_str = gen_args.text_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    tokenizer = T5TokenizerFast.from_pretrained(gen_args.text_t5_model)

    model = DualObjectFMRItoT5(
        fmri_dim=gen_args.text_fmri_dim,
        t5_model_name=gen_args.text_t5_model,
        num_tokens=gen_args.text_num_tokens
    )
    state_dict = torch.load(gen_args.text_model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset, batch_size=gen_args.text_batch_size, shuffle=False)
    max_samples = gen_args.text_max_samples if gen_args.text_max_samples > 0 else None

    results = []
    processed = 0
    progress = tqdm(dataloader, desc="Generating text from test fMRI", leave=False)

    with torch.no_grad():
        for batch in progress:
            if max_samples is not None and processed >= max_samples:
                break

            fmri_vectors = batch["fmri"].to(device)
            generated_ids = model.generate_sentences(fmri_vectors, max_length=gen_args.text_max_length)
            decoded_batch = decode_generated_batch(generated_ids, tokenizer)

            for local_idx, decoded_sentences in enumerate(decoded_batch):
                sample_idx = processed + local_idx
                results.append({
                    "sample_idx": sample_idx,
                    "generated": decoded_sentences
                })

            processed += fmri_vectors.size(0)

    if max_samples is not None:
        results = results[:max_samples]

    if gen_args.text_output_json:
        output_dir = os.path.dirname(gen_args.text_output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(gen_args.text_output_json, "w") as f:
            json.dump(results, f, indent=2)

    return results




if __name__ == "__main__":
    gen_args = parse_text_generation_args()

    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = GeneratorPipe(model_id, dtype=torch.float16, device="cuda:7")
    negative_prompt = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

    _, _, test_data = load_all_data()
    text_results = generate_text_from_test_data(test_data, gen_args)
    result_img = []
    for sample in text_results:
        prompt, direction = decoded_sentences_to_prompt_and_direction(sample["generated"])

        image = pipe(prompt, negative_prompt,
                      batch_size=1,  # batch size
                      num_inference_steps=gen_args.num_inference_steps,  # sampling step
                      height=gen_args.img_height,
                      width=gen_args.img_width,
                      end_steps=gen_args.end_steps,
                      base_ratio=gen_args.base_ratio,
                      direction=direction)
        result_img.append(image[0])


    _, gt_imgs, _, _ = load_data_lists("test", gen_args.save_dir)
    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    img_tensor = torch.stack([to_tensor(im) for im in result_img], dim=0)
    gt_tensor = torch.stack([to_tensor(im) for im in gt_imgs], dim=0)

    mean_lpips = eval_lpips(img_tensor, gt_tensor)
    print(f"LPIPS: {mean_lpips:.4f}")


    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights),
                                               return_nodes=['avgpool']).to("cuda:0")
    inception_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize((342, 342), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    all_per_correct = two_way_identification(result_img, gt_imgs, inception_model, preprocess, 'avgpool')
    print(f"Inception V3 identification: {all_per_correct:.4f}")

    model, preprocess, device = load_openclip_model(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")

    pairwise_score = two_way_identification_openclip(all_recons=result_img, all_images=gt_imgs, model=model,
        preprocess=preprocess, device=device, return_avg=True)
    print("Clip identification:", pairwise_score)

    mean_pixcor = eval_pixcor(img_tensor, gt_tensor)
    print(f"Pixel Correlation: {mean_pixcor:.4f}")
    mean_ssim = eval_ssim(img_tensor, gt_tensor)
    print(f"SSIM: {mean_ssim:.4f}")
