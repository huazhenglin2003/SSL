import datetime
import os
import json
import random
import argparse
import torch
from math import ceil
from tqdm import tqdm
from sae import Sae
from datasets import *
from utils import *


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chair", help="Dataset name (options: chair, pope)")
    parser.add_argument("--model_path", type=str, default="/path/to/model_cache/models--meta-llama--Llama-3.2-11B-Vision-Instruct", help="Path to the pretrained model")
    parser.add_argument("--output_file", type=str, default="test/test.json", help="Path to save the output results (default: test/test.json)")
    parser.add_argument("--num_chunks", type=int, default=8, help="Number of chunks to split the dataset into")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the current chunk to process")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate (default: 512)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Gamma value for steering strength (default: 0.2)")
    parser.add_argument("--layer", type=int, default=24, help="Layer indices to apply steering to.")
    parser.add_argument("--subset", type=str, default="adversarial", choices=["popular", "adversarial", "random"], help="Subset of dataset to use (default: popular)")

    return parser.parse_args()

                
def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    random.seed(args.seed)

    # Load model
    device = torch.device(args.device)
    processor, model, template, ans_start, num_img_tokens, image_start = get_model(args.model_path, device, "eager")

    # Load dataset
    if args.dataset == "chair":
        dataset = CHAIRBench(500)
        print(f"seed: {args.seed}")
    elif args.dataset == "pope":
        dataset = POPE("pope_" + args.subset)
        print(f"dataset: pope_{args.subset}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    chunk_size = ceil(len(dataset.all) / args.num_chunks)

    output_file = open(args.output_file, "w")

    generate_kwargs = {"num_beams": 1, 
                       "max_new_tokens": args.max_new_tokens,
                       "do_sample": False}
    
    # TODO update sae path if needed
    sae = Sae.load_from_disk(
        r"/data/sae/llama3-llava-next-8b-hf-sae-131k/model.layers.24/",
        device=device
    )
    hall_index = 36992
    non_hall_index = 47230
    W_n_loaded = sae.W_dec.clone()[non_hall_index, :].detach().view(1, 1, -1)
    W_loaded = sae.W_dec.clone()[hall_index, :].detach().view(1, 1, -1)
    print(f"SSL Layer: {args.layer}")
    
    if hasattr(model.language_model, "layers"):
        hooked_module = model.language_model.layers[args.layer]
    elif hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        hooked_module = model.language_model.model.layers[args.layer]
    else:
        raise ValueError("Unsupported model type for hooking")

    handle = ssl(
        d_hall=W_loaded,
        d_non_hall=W_n_loaded,
        hooked_module=hooked_module,
        image_start=image_start,
        num_img_tokens=num_img_tokens,
        gamma=args.gamma,
        device=device
    )
       
    start = datetime.datetime.now()    
    for qid in tqdm(dataset.all[args.chunk_idx*chunk_size:(args.chunk_idx+1)*chunk_size], ncols=100):

        # prepare for inputs
        img, qu, gt = dataset[qid]
        prompt = template.format(question=qu)
        inputs = processor(text=prompt, images=img, return_tensors="pt")

        inputs = {key: value.to(device) for key, value in inputs.items()}
            
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs,  output_scores=True, return_dict_in_generate=True)

        gen_answer = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        gen_answer = gen_answer.split(ans_start)[1].strip() if ans_start is not None else gen_answer.strip()
        
        probs = outputs.scores[0][0].softmax(-1)
        ans_conf = probs.topk(1)[0].cpu().numpy()[0]
        ans_conf = str(ans_conf*100)[:5]

        output_file.write(json.dumps({"question_id": qid, "text": gen_answer, "gt": gt, "ans_conf": ans_conf, "question": qu})+"\n")

    output_file.close()
    end = datetime.datetime.now()
    elapsed = (end - start).seconds
    print(f"inference_time_s: {elapsed}")

if __name__ == "__main__":
    args = argument_parser()
    main(args)