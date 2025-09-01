# Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation

Official implementation of our paper:
**"Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation"**, accepted to **EMNLP 2025 Findings**.

📄 Paper: [https://arxiv.org/pdf/2505.16146](https://arxiv.org/pdf/2505.16146)

---

## Installation

```bash
conda create -n ssl python=3.10.14
conda activate ssl
pip install -r requirements.txt
```

> ⚠️ **Note**: The default `requirements.txt` is configured for *LLaVA-NeXT-8B*, *LLaVA-1.5-7B* and *InstructBLIP-7B*. If you plan to use **Llama-3.2-11B-Vision-Instruct** for inference, you must **upgrade `transformers` to version `4.53.0`**:
```bash
pip install --upgrade "transformers==4.53.0"
```

---

## Dataset and Model Preparation

1. **Datasets**. Download [MSCOCO](https://cocodataset.org/#download) and organize the files under `./data` as follows:

   ```
   ├── coco
   │     ├── val2014
   │     └── annotations
   │           ├── captions_val2014.json
   │           └── instances_val2014.json
   └── pope
         └── coco
               ├── coco_pope_popular.json
               ├── coco_pope_random.json
               └── coco_pope_adversarial.json
   ```

2. **Large Vision-Language Models (LVLMs)**. Download the following LVLMs and update `model_dir` in the `.sh` scripts if needed:

   * [LLaVA-NeXT-8B](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/main)
   * [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)
   * [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/tree/main)
   * [InstructBLIP-7B](https://huggingface.co/Salesforce/instructblip-vicuna-7b/tree/52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92)

3. **Sparse Autoencoder (SAE).** This work uses the SAE provided by **lmms-lab**:
   [llama3-llava-next-8b-hf-sae-131k](https://huggingface.co/lmms-lab/llama3-llava-next-8b-hf-sae-131k/tree/main).
   After downloading, organize it under:

   ```
   ├── data
         └── sae
               └── llama3-llava-next-8b-hf-sae-131k
   ```

---

## Inference and Evaluation

Run inference using the provided shell scripts:

```bash
# General syntax
bash infer_script.sh

# Example runs
bash scripts/infer_chair.sh
bash scripts/infer_pope.sh
```

👉 **Note**:

* You can set the target LVLM and adjust **gamma** and **layer** hyperparameters in each `.sh` script.
* `num_chunks` controls parallel GPU usage (default: 8).

## Citations

If you find this work useful, please cite:

```bibtex
@misc{hua2025steeringlvlmssparseautoencoder,
      title={Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation}, 
      author={Zhenglin Hua and Jinghan He and Zijun Yao and Tianxu Han and Haiyun Guo and Yuheng Jia and Junfeng Fang},
      year={2025},
      eprint={2505.16146},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.16146}
}
```