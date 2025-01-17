# Finetuning Llama30b on a 24G card
Here are some instructions on how to finetune Llama30B on a 24G card with the Alpaca dataset.

_Note: Things are finally starting to stabilize a bit in the various repos.  But let me know if you run into snags with these instructions and I'll try to update them._

Three key things allow us to finetune Llama30B on a 24G card 
1) Lora adapters (using huggingface's peft)
2) int4 quantization
3) Cuda/Triton extensions for required int4 operations (forward and autograd operations).

Peft (using Lora finetuning) freezes the base model weights and adds much smaller lower rank tensors to the model that can be finetuned. This greatly reduces the amount of memory needed to train the model since there are far fewer trainable parameters.

Int4 quantization allows us to reduce the memory needed for the weights by half over what can be achieved with int8 quantization.

## Obtain the original Llama weights
I was able to get them from the torrent listed in this open pull request on the llama repo
https://github.com/facebookresearch/llama/pull/73/files

## Convert the Llama weights into the hugging face transformer model format and generate a quantized version of the model that uses int4 weights

https://github.com/qwopqwop200/GPTQ-for-LLaMa


``` bash
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c -y
pytorch -c nvidia
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt

```

Convert the 30B llama weights to hugging face (hf) format

``` bash
python convert_llama_weights_to_hf.py --input_dir /path/to/Llama/weights/ --model_size 30B --output_dir ./llama-hf
```

Quantize the hf model to int4 (Specifying GPU to use)

``` bash
CUDA_VISIBLE_DEVICES=1 python llama.py ./llama-hf/llama-30b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save_safetensors llama30b-4bit-128g.safetensors
```

## Finetune the 30b int4 quantized model on the alpaca dataset  
My repo fork contains just a few small things on top of the upstream repo.

* Contains the alpaca cleaned dataset for convenience
* Enables using wandb (optional) for logging the training run
* reenables eval during training
* Doubles the alpaca dataset size by inverting the output/instructions.  This is a hack to get more data for training.
* changed load_dataset calls so that the finetuning datasets can be loaded from HF hub
* patch to allow triton backend to work with triton 2.0 (from pypi)

``` bash
cd ..
conda create --name alpaca_lora_4bit python=3.9 -y
conda activate alpaca_lora_4bit
git clone https://github.com/johnrobinsn/alpaca_lora_4bit.git
cd alpaca_lora_4bit
pip install torch
pip install -r requirements.txt
```
Finetune with alpaca dataset (instruction flipping enabled)
_Note: Estimated time is 120 hours on my Titan RTX.  3 epochs._

``` bash
CUDA_VISIBLE_DEVICES=1 python finetune.py --ds_type alpaca --groupsize 128 --grad_chckpt --llama_q4_config_dir ../GPTQ-for-LLaMa/llama-hf/llama-30b/ --llama_q4_model ../GPTQ-for-LLaMa/llama30b-4bit-128g.safetensors --wandb johnrobinsn/alpaca-cleaned
```
Finetune with gpt4all dataset instead
_Note: Estimated time is 600 hours on my Titan RTX.  3 epochs._

``` bash
python finetune.py --ds_type gpt4all --groupsize 128 --grad_chckpt --llama_q4_config_dir ../GPTQ-for-LLaMa/llama-hf/llama-30b/ --llama_q4_model ../GPTQ-for-LLaMa/llama30b-4bit-128g.safetensors --wandb nomic-ai/gpt4all_prompt_generations
```