from collections import defaultdict
import torch.nn.functional as F
import torch

class HookManager:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = defaultdict(lambda : [])
        self.early_exit_words = []

    def attn_hook(self, module, input, output):
        # print(f"Attention Hook")
        pass

    def early_exit_hook(self, module, input, output):
        norm_output = self.model.model.norm(output[0])
        logits = self.model.lm_head(norm_output)
        logits = logits[:, -1, :]
        _, max_index = torch.max(logits, dim=-1)
        # word = self.tokenizer.decode(max_index.tolist()
        word = max_index
        self.early_exit_words.append(word)
        # print(f"Early exit hook: {word}")

    def register_hooks(self, layers: list[int], type_hook: str) -> None:
        for layer in layers:
            self.register_hook(layer, type_hook)

    def register_hook(self, layer_i: int, type_hook: str):
        if type_hook == "early_exit":
            layer  = self.model.model.layers[layer_i]
            self.hooks[type_hook].append(layer.register_forward_hook(self.early_exit_hook))
        elif type_hook == "attn":
            layer = self.model.model.layers[layer_i].self_attn
            self.hooks[type_hook].append(layer.register_forward_hook(self.attn_hook))
        else:
            raise Exception("Not a valid hook type")

    def remove_hooks(self):
        for t in self.hooks.keys():
            for h in self.hooks[t]:
                h.remove()
        self.hooks = defaultdict(lambda : [])