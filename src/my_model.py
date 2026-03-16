import os

import numpy as np
import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer


class DualObjectFMRItoT5(nn.Module):
    def __init__(self, fmri_dim=15724, t5_model_name="t5-base", num_tokens=5):
        super().__init__()
        # Load pretrained T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=True)

        t5_dim = self.t5_model.config.d_model
        self.num_tokens = num_tokens
        self.num_objects = 2  # We have 2 objects
        self.num_sentences_per_object = 2  # 1 sentences per object, or 2
        self.seq_len = 5
        self.fmri_to_objects_mlp = nn.Sequential(
            nn.Linear(fmri_dim, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * 10 * 1024)
        )

        self.mlp_modules = nn.ModuleList([
            ObjectToSentenceMLP(input_dim=10 * 1024, output_dim=t5_dim, seq_len=self.num_tokens, num_sentences=2)
            for _ in range(self.num_objects)
        ])



    def forward(self, fmri_vector, labels=None):
        batch_size = fmri_vector.shape[0]

        all_object_embeddings = self.fmri_to_objects_mlp(fmri_vector)
        all_object_embeddings = all_object_embeddings.view(batch_size, self.num_objects, 10 * 1024)

        all_outputs = []
        for obj_idx in range(self.num_objects):
            # Get embeddings for current object
            obj_embedding = all_object_embeddings[:, obj_idx, :]
            t5_inputs = self.mlp_modules[obj_idx](obj_embedding)
            for sent_idx in range(self.num_sentences_per_object):
                # Get embedding for current sentence
                sent_embedding = t5_inputs[:, sent_idx]
                full_embedding = sent_embedding

                # Create attention mask for T5
                encoder_attention_mask = torch.ones(
                    batch_size, full_embedding.shape[1], device=full_embedding.device
                )

                global_sent_idx = obj_idx * self.num_sentences_per_object + sent_idx

                if labels is None:
                    # For inference
                    decoder_input_ids = torch.full(
                        (batch_size, 1),
                        self.t5_model.config.decoder_start_token_id,
                        dtype=torch.long,
                        device=full_embedding.device
                    )
                    outputs = self.t5_model(
                        inputs_embeds=full_embedding,
                        attention_mask=encoder_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        return_dict=True
                    )
                else:
                    # For training with labels
                    current_labels = labels[global_sent_idx] if isinstance(labels, list) else labels
                    outputs = self.t5_model(
                        inputs_embeds=full_embedding,
                        attention_mask=encoder_attention_mask,
                        labels=current_labels,
                        return_dict=True
                    )

                all_outputs.append(outputs)

        return all_outputs

    def generate_sentences(self, fmri_vector, max_length=30):
        batch_size = fmri_vector.shape[0]

        # Map fMRI to object embeddings
        all_object_embeddings = self.fmri_to_objects_mlp(fmri_vector)
        all_object_embeddings = all_object_embeddings.view(batch_size, self.num_objects, 10 * 1024)

        all_generated_ids = []

        for obj_idx in range(self.num_objects):
            # Get embeddings for current object
            obj_embedding = all_object_embeddings[:, obj_idx, :]

            t5_inputs = self.mlp_modules[obj_idx](obj_embedding)  # [batch_size, 2, 5, t5_dim]

            # Process each sentence representation through T5
            for sent_idx in range(self.num_sentences_per_object):
                sent_embedding = t5_inputs[:, sent_idx]

                full_embedding = sent_embedding
                # Create attention mask for T5
                encoder_attention_mask = torch.ones(
                    batch_size, full_embedding.shape[1], device=full_embedding.device
                )
                # Generate text for this sentence
                generated_ids = self.t5_model.generate(
                    inputs_embeds=full_embedding,
                    attention_mask=encoder_attention_mask,
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

                all_generated_ids.append(generated_ids)

        return all_generated_ids


    def freeze_t5(self):
        """Freeze the entire T5 model"""
        for param in self.t5_model.parameters():
            param.requires_grad = False


    def unfreeze_all(self):
        """Unfreeze the entire T5 model"""
        for param in self.t5_model.parameters():
            param.requires_grad = True


class ObjectToSentenceMLP(nn.Module):
    def __init__(self, input_dim=10 * 1024, output_dim=768, seq_len=5, num_sentences=2):
        super(ObjectToSentenceMLP, self).__init__()

        self.num_sentences = num_sentences
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Create a separate MLP for each sentence type
        self.sentence_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, output_dim * seq_len)
            ) for _ in range(num_sentences)
        ])

    def forward(self, object_embedding):
        batch_size = object_embedding.size(0)

        sentence_outputs = []
        for i in range(self.num_sentences):
            output = self.sentence_mlps[i](object_embedding)
            output = output.view(batch_size, self.seq_len, self.output_dim)

            sentence_outputs.append(output)
        # Stack sentence representations
        sentence_outputs = torch.stack(sentence_outputs, dim=1)

        return sentence_outputs

