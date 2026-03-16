import copy
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from my_model import DualObjectFMRItoT5
from create_data import load_all_data


def train_fmri_to_text_model(model, train_dataloader, val_dataloader, factor, patience, num_epochs=40, stage_1_epochs=30,
                             learning_rate=1e-5, device="cuda", tokenizer=None,
                             model_save_dir="/model_ckpt"):
    stages = [
        {"name": "Stage 1: Adapters only", "epochs": stage_1_epochs, "lr": learning_rate},
        {"name": "Stage 2: All layers", "epochs": num_epochs - stage_1_epochs, "lr": learning_rate/50}
    ]

    model = model.to(device)

    global_epoch = 0
    for stage_idx, stage in enumerate(stages):
        print(f"\n===== Starting {stage['name']} =====")
        if stage_idx == 0:
            # Stage 1: train adapters
            model.freeze_t5()
        elif stage_idx == 1:
            # Stage 2: Unfreeze everything
            model.unfreeze_all()

        # Set up optimizer with different learning rates for adapter and T5
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=stage["lr"])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(stage["epochs"]):
            global_epoch += 1

            # Training phase
            model.train()
            train_loss = 0
            train_bar = tqdm(train_dataloader,
                             desc=f"Epoch {global_epoch}/{num_epochs} [Train]")

            for batch in train_bar:
                fmri_vectors = batch["fmri"].to(device)
                labels = [label.to(device) for label in batch["labels"]]

                outputs = model(fmri_vectors, labels=labels)
                # loss = outputs.loss
                loss = sum(output.loss for output in outputs) / len(outputs)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_bar.set_postfix(loss=train_loss / (train_bar.n + 1))

            avg_train_loss = train_loss / len(train_dataloader)

            # Validation phase
            model.eval()
            val_loss = 0
            # generated_sentences = []
            # reference_sentences = []

            val_bar = tqdm(val_dataloader,
                           desc=f"Epoch {global_epoch}/{num_epochs} [Val]")

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_bar):
                    fmri_vectors = batch["fmri"].to(device)
                    labels = [label.to(device) for label in batch["labels"]]

                    outputs = model(fmri_vectors, labels=labels)
                    batch_loss = sum(output.loss.item() for output in outputs) / len(outputs)
                    val_loss += batch_loss

                    # Generate text for a few examples
                    # if batch_idx < 2:  # Limit to first 2 batches to save time
                    #     for i in range(min(2, fmri_vectors.size(0))):  # Take first 2 examples from batch
                    #         single_fmri = fmri_vectors[i:i + 1]
                    #
                    #         # Generate sentences for each segment
                    #         generated_ids = model.generate_sentences(single_fmri, max_length=50)
                    #
                    #         # Decode generated IDs to text
                    #         sample_generated = []
                    #         for ids in generated_ids:
                    #             decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
                    #             sample_generated.append(decoded)
                    #
                    #         # Get reference sentences
                    #         sample_reference = list(batch["text"][i])
                    #
                    #         generated_sentences.append(sample_generated)
                    #         reference_sentences.append(sample_reference)

                    val_bar.set_postfix(loss=val_loss / (val_bar.n + 1))

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {global_epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # for i in range(min(3, len(generated_sentences))):
            #     print(f"\nExample {i + 1}:")
            #     for j in range(len(generated_sentences[i])):
            #         print(f"Segment {j + 1}:")
            #         print(f"  Generated: {generated_sentences[i][j]}")
            #         print(f"  Reference: {reference_sentences[i][j]}")
            #     print("-" * 80)

            scheduler.step(avg_val_loss)

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model with validation loss: {best_val_loss:.4f}")
                torch.save(best_model_state, f"{model_save_dir}/cur_best_large_lr.pt")
                print("Training complete and model saved!")

    # Load the best model state
    # model.load_state_dict(best_model_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=True)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32)

    parser.add_argument(
        "--fmri_dim",
        type=int,
        default=15724)

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=6)

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=80)

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7)
    parser.add_argument("--stage_1_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--factor", type=float, default=0.5)


    args = parser.parse_args()
    model_save_dir = args.model_save_dir
    batch_size = args.batch_size
    fmri_dim = args.fmri_dim
    num_tokens = args.num_tokens
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    stage_1_epochs = args.stage_1_epochs
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")

    train_dataset, val_dataset, _ = load_all_data()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DualObjectFMRItoT5(fmri_dim=fmri_dim, t5_model_name="t5-base", num_tokens=num_tokens)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model = train_fmri_to_text_model(
        model, train_dataloader, val_dataloader, factor=args.factor, patience=args.patience,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device, stage_1_epochs=stage_1_epochs ,tokenizer=tokenizer,
        model_save_dir=model_save_dir)

    # Save
    torch.save(trained_model.state_dict(), f"{model_save_dir}/model.pt")
    print("Training complete and model saved!")


if __name__ == "__main__":
    main()
