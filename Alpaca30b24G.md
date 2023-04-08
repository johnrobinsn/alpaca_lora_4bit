# Finetuning llama30b on a 24G card

Things are finally starting to stabilize a bit in the various repos.  But let me know if you run into snags with these instructions and I'll try update them.  The key things that allow us to finetune Llama30B on a 24G card this are lora adapters (using huggingface's peft) and int4 quantization.

Peft using Lora finetuning allow us to freeze the base model weights and add much smaller lower rank tensors to the model that can be finetuned this greatly reduces the memory needed to train the model since there are far fewer trainable parameters.

Int4 quantization allows us to reduce the memory needed for the weights by half over what can be achieved with int8 quantization.

Here are some instructions on how to finetune Llama30B on a 24G card with the Alpaca dataset.

## Obtain the original Llama weights
I was able to get them from the torrent listed in this open pull request on the llama repo
https://github.com/facebookresearch/llama/pull/73/files

## Convert the Llama weights into the hugging face transformer model format and generate a quantized version of the model that uses int4 weights

https://github.com/qwopqwop200/GPTQ-for-LLaMa


``` bash
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt

```

Convert the 30B llama weights to hugging face format

``` bash
python convert_llama_weights_to_hf.py --input_dir /path/to/Llama/weights/ --model_size 30B --output_dir ./llama-hf
```

Quantize the hf model to int4 (Specifying GPU to use)

``` bash
CUDA_VISIBLE_DEVICES=1 python llama.py ./llama-hf/llama-30b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save_safetensors llama30b-4bit-128g.safetensors
```

## Finetune the 30b int4 quantized model on the alpaca dataset  
My repo fork contains just a few small things ontop of the upstream repo.

* Contains the alpaca cleaned dataset
* Enables using wandb for logging the training run
* reenables eval during training
* Doubles the alpaca dataset size by inverting the output/instructions.  This is a hack to get more data for training.

``` bash
cd ..
conda create --name alpaca_lora_4bit python=3.9 -y
conda activate alpaca_lora_4bit
git clone https://github.com/johnrobinsn/alpaca_lora_4bit.git
cd alpaca_lora_4bit
pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=1 python finetune.py --ds_type alpaca --groupsize 128 --grad_chckpt --llama_q4_config_dir ../gptq/llama-hf/llama-30b/ --llama_q4_model ../gptq/llama30b-4bit-128g.safetensors ./alpaca_data_cleaned.json --wandb
```
