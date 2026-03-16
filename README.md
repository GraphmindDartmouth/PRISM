## PRISM
Codes for paper Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI (PRISM).

## Introduction
We propose PRISM, a model that Projects fMRI sIgnals into a Structured text space as an interMediate representation for visual stimuli reconstruction. It includes an object centric diffusion module that generates images by composing individual objects to reduce object detection errors, and an attribute relationship search module that automatically identifies key attributes and relationships that best align with the neural activity

## Requirements

### Clone the repository:
```bash
git clone https://github.com/GraphmindDartmouth/PRISM.git
```

### Install the required packages using conda:
```bash
  cd src
  conda env create -f env.yml
   ```
Experiments are carried out on a NVIDIA L40 with CUDA Version 12.2.

### NSD data:
Agree to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form](https://forms.gle/xue2bCdM9LaFNMeb7). Then download the NSD data.

## Quick Start
Run the following command to execute the entire pipeline, which includes keyword generation, structured text generation, model training, and image generation. Make sure to replace `{api key}` with your actual OpenAI API key.
```bash
. quick_start.sh
```

## Step-by-Step Instructions
```bash
python keyword_generator.py --openai_api_key {api key}
   ```
This code generates keywords that align with the fmri data, which are then used to guide the structured text generation process. 

```bash
python get_gpt_data.py  --openai_api_key {api key} --data_split train
python get_gpt_data.py  --openai_api_key {api key} --data_split val
python get_gpt_data.py  --openai_api_key {api key} --data_split test
```
This code generates structured text data for training, validation, and testing. The structured text is generated based on the keywords obtained from the previous step and is designed to align with the neural activity patterns observed in the fMRI data.

```bash
python train_model.py 
```
This code trains the proposed model using the generated structured text as supervision.

```bash
python run_gen.py
```
This code runs the trained model to generate images based on the fMRI data. 

## Logs
The reference logs are located in the logs/ folder.

## Contact
If you have any questions, suggestions, or bug reports, please contact
```
zheng.huang.gr@dartmouth.edu
```

