class HookManager:

    def __init__(self):
        self.hooks = []

    def hook(self, module, input, output):
        print(f"Hidden states shape for {module}: {output[0].shape}")

    def register_hook(self, model, layer_i):
        layer  = model.encoder.layers[layer_i]
        self.hooks.append(layer.register_forward_hook(self.hook))
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()