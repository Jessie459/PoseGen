import argparse
import json
import os
import warnings

import torch
from diffsynth.prompters import WanPrompter
from PIL import Image
from wan.utils.prompt_extend import QwenPromptExpander

warnings.filterwarnings("ignore")

TOKEN_DICT = {"woman": 27502, "girl": 15146, "man": 621, "boy": 18942}


class ExtendedWanPrompter(WanPrompter):
    def get_ids(self, prompt, positive=True):
        prompt = self.process_prompt(prompt, positive=positive)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    prompt_path = args.prompt_path
    if not prompt_path:
        prompt_path = os.path.join(args.output_dir, "prompt.txt")

        # Create the prompt expander
        prompt_expander = QwenPromptExpander(os.environ["QWEN25_VL_7B_INSTRUCT_PATH"], is_vl=True, device="cuda")

        # Load the reference image
        image = Image.open(args.image_path).convert("RGB")

        # Generate the text prompt
        prompt = ""
        prompt_output = prompt_expander(prompt, tar_lang="en", image=image, seed=args.seed)
        if not prompt_output.status:
            print(f"Extending prompt failed: {prompt_output.message}")
        else:
            prompt = prompt_output.prompt
        print(f"Generated prompt:\n{prompt}")

        # Save the text prompt
        os.makedirs(args.output_dir, exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(prompt)

    # Load the text prompt
    with open(prompt_path, "r") as f:
        prompt_list = f.readlines()
    if len(prompt_list) == 1:
        prompt = prompt_list[0]
    else:
        prompt = max(prompt_list, key=len)
    print(f"Loaded prompt:\n{prompt}")

    # Get token ids
    prompter = ExtendedWanPrompter()
    prompter.fetch_tokenizer(os.path.join(os.environ["POSEGEN_CKPT_PATH"], "google/umt5-xxl"))

    ids = prompter.get_ids(prompt, positive=True)
    ids = ids.squeeze(0).numpy().tolist()

    valid = False
    for keyword in TOKEN_DICT.keys():
        if keyword in prompt:
            print(f"Keyword: {keyword}")
            print(f"Keyword token position: {ids.index(TOKEN_DICT[keyword])}")
            valid = True
            break
    if not valid:
        raise RuntimeError(
            f"""The generated prompt does not contain any keywords (woman, girl, man, boy). 
            Please manually modify the prompt at {prompt_path}, 
            and then run `python generate_prompt.py --prompt_path {prompt_path}`"""
        )

    # Save token ids
    with open(prompt_path.replace(".txt", ".json"), "w") as f:
        json.dump(ids, f)


if __name__ == "__main__":
    main()
