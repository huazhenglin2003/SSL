import torch

from transformers.models.instructblip.modeling_instructblip import *

from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration


def get_model(model_path, device, attn_implementation="eager"):
    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if "instructblip" in model_path:
        processor = InstructBlipProcessor.from_pretrained(model_path, use_fast=False)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        template = "{question}"
        ans_start = None
        num_img_tokens = model.config.num_query_tokens
        image_start = 0
        
    elif "llava-next" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{question}"},
                    {"type": "image"},
                ],
            },
        ]
        template = processor.apply_chat_template(conversation, add_generation_prompt=True)
        ans_start = "assistant"
        num_img_tokens = 576
        image_start = ((processor(text=template, return_tensors="pt")['input_ids']) == model.config.image_token_index).nonzero()[0, 1].item() + 1  # skip <image>
        
    elif "Llama-3.2" in model_path:
        from transformers import MllamaForConditionalGeneration
        
        processor = AutoProcessor.from_pretrained(model_path)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "{question}"}     
                ]
            }
        ]
        template = processor.apply_chat_template(messages, add_generation_prompt=True)
        ans_start = "assistant"
        num_img_tokens = 1600
        image_start = 6 + 1  # skip <image>
        
    else:
        model_kwargs["attn_implementation"] = "eager"
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        template = "USER: <image>\n{question} ASSISTANT:"
        ans_start = "ASSISTANT:"
        num_img_tokens = 576
        image_start = ((processor(text=template, return_tensors="pt")['input_ids']) == model.config.image_token_index).nonzero()[0, 1].item() + 1  # skip <image>

    model.to(device)

    return processor, model, template, ans_start, num_img_tokens, image_start


def ssl(
    d_hall: torch.tensor,
    d_non_hall: torch.tensor,
    image_start: int,
    num_img_tokens: int, 
    hooked_module: torch.nn.Module,
    gamma: float,
    device
):  
    d_hall = d_hall.to(device)
    d_non_hall = d_non_hall.to(device) 

    def hook(module: torch.nn.Module, _, outputs):
        nonlocal d_hall, d_non_hall
        with torch.no_grad():
            if isinstance(outputs, tuple):
                unpack_outputs = list(outputs)
            else:
                unpack_outputs = list(outputs)
            
            if unpack_outputs[0].shape[1] != 1:
                x_img = outputs[0][:, image_start:image_start+num_img_tokens, :]
                x_norm = x_img.norm(dim=-1, keepdim=True)
                d_norm = d_non_hall.norm() + 1e-6
                alpha_img = gamma * x_norm / d_norm

                unpack_outputs[0][:, image_start:image_start+num_img_tokens, :] = (
                    x_img + alpha_img * d_non_hall.to(outputs[0].dtype)
                )

            else:
                x_gen = outputs[0]
                x_norm = x_gen.norm()
                d_norm = d_hall.norm() + 1e-6
                alpha_gen = gamma * x_norm / d_norm

                unpack_outputs[0] = x_gen - alpha_gen * d_hall.to(outputs[0].dtype)
                
        return tuple(unpack_outputs) if isinstance(outputs, tuple) else unpack_outputs[0]
    
    return hooked_module.register_forward_hook(hook)