import hashlib
import os
from contextlib import contextmanager

import torch
from safetensors import safe_open


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(
                        search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    )
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


class GeneralLoRAFromPeft:
    def __init__(self):
        pass

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def match(self, model: torch.nn.Module, state_dict_lora):
        lora_name_dict = self.get_name_dict(state_dict_lora)
        model_name_dict = {name: None for name, _ in model.named_parameters()}
        matched_num = sum([i in model_name_dict for i in lora_name_dict])
        if matched_num == len(lora_name_dict):
            return "", ""
        else:
            return None

    def fetch_device_and_dtype(self, state_dict):
        device, dtype = None, None
        for name, param in state_dict.items():
            device, dtype = param.device, param.dtype
            break
        computation_device = device
        computation_dtype = dtype
        # if computation_device == torch.device("cpu"):
        #     if torch.cuda.is_available():
        #         computation_device = torch.device("cuda")
        if computation_dtype == torch.float8_e4m3fn:
            computation_dtype = torch.float32
        return device, dtype, computation_device, computation_dtype

    def load(self, model, state_dict_lora, alpha=1.0):
        state_dict_model = model.state_dict()
        device, dtype, computation_device, computation_dtype = self.fetch_device_and_dtype(state_dict_model)
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name in lora_name_dict:
            weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
            weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
            weight_patched = weight_model + weight_lora
            state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
        print(f"[{self.__class__.__name__}] {len(lora_name_dict)} tensors are updated.")
        model.load_state_dict(state_dict_model)


def _get_name_dict(lora_state_dict):
    lora_name_dict = {}
    for key in lora_state_dict:
        if ".lora_B." not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        target_name = ".".join(keys)
        lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
    return lora_name_dict


def merge_lora_weights(ckpt_path, lora_alpha=1.0):
    lora_state_dict = {}
    base_state_dict = {}

    prefix = "pipe.dit."

    if os.path.isdir(ckpt_path):
        ckpt_files = sorted([x for x in os.listdir(ckpt_path) if x.endswith(".bin")])
        for ckpt_file in ckpt_files:
            partial_ckpt_path = os.path.join(ckpt_path, ckpt_file)
            print(f"Loading checkpoint from: {partial_ckpt_path}")
            state_dict = load_state_dict(partial_ckpt_path)
            for name, param in state_dict.items():
                if "lora" in name:
                    lora_state_dict[name[len(prefix):]] = param.clone()
                else:
                    base_state_dict[name[len(prefix):].replace(".base_layer", "")] = param.clone()
            del state_dict
    else:
        print(f"Loading checkpoint from: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path)
        for name, param in state_dict.items():
            if "lora" in name:
                lora_state_dict[name[len(prefix):]] = param.clone()
            else:
                base_state_dict[name[len(prefix):].replace(".base_layer", "")] = param.clone()
        del state_dict

    for name, param in base_state_dict.items():
        device, dtype = param.device, param.dtype
        break

    lora_name_dict = _get_name_dict(lora_state_dict)
    for name in lora_name_dict:
        weight_up = lora_state_dict[lora_name_dict[name][0]].to(device=device, dtype=dtype)
        weight_down = lora_state_dict[lora_name_dict[name][1]].to(device=device, dtype=dtype)
        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = lora_alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = lora_alpha * torch.mm(weight_up, weight_down)
        base_state_dict[name] = base_state_dict[name] + weight_lora

    return base_state_dict
