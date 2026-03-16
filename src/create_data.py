import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer
from new_nsd_data import load_data_lists
import argparse


class FMRITextDataset(Dataset):
    def __init__(self, fmri_vectors, text_descriptions, tokenizer, max_length=128):

        self.fmri_vectors = fmri_vectors
        self.text_descriptions = text_descriptions

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.fmri_vectors)

    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_vectors[idx], dtype=torch.float)

        sentences = self.text_descriptions[idx]
        tokenized_sentences = []

        for sentence in sentences:
            tokenized_sentence = self.tokenizer(
                sentence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tokenized_sentences.append(tokenized_sentence.input_ids.squeeze(0))


        return {
            "fmri": fmri,
            "labels": tokenized_sentences,
            "text": sentences
        }



def load_all_data(save_dir="data"):
    split = "train"
    save_dir = save_dir
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")


    fmri_train, img_train, caption_train, coco_id_train = load_data_lists(split, save_dir)
    print("Data loaded! Length: ", len(fmri_train), len(img_train), len(coco_id_train))
    structured_response_train = np.load(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz', allow_pickle=True)
    structured_response_train = structured_response_train['structured_response']
    assert len(fmri_train) == len(img_train) == len(coco_id_train) == len(structured_response_train)
    train_data = FMRITextDataset(fmri_train, structured_response_train, tokenizer)

    split = "val"
    fmri_val, img_val, caption_val, coco_id_val = load_data_lists(split, save_dir)
    print("Data loaded! Length: ", len(fmri_val), len(img_val), len(coco_id_val))
    structured_response_val = np.load(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz', allow_pickle=True)
    structured_response_val = structured_response_val['structured_response']
    assert len(fmri_val) == len(img_val) == len(coco_id_val) == len(structured_response_val)
    val_data = FMRITextDataset(fmri_val, structured_response_val, tokenizer)

    split = "test"
    fmri_test, img_test, caption_test, coco_id_test = load_data_lists(split, save_dir)
    print("Data loaded! Length: ", len(fmri_test), len(img_test), len(coco_id_test))
    structured_response_test = np.load(f'{save_dir}/brain_data_{split}_final_cc_rewrite.npz', allow_pickle=True)
    structured_response_test = structured_response_test['structured_response']
    assert len(fmri_test) == len(img_test) == len(coco_id_test) == len(structured_response_test)
    test_data = FMRITextDataset(fmri_test, structured_response_test, tokenizer)


    return train_data, val_data, test_data

