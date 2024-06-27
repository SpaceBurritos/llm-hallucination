from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from hooks.hook import HookManager

def main():
    hm = HookManager()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", 
                                                quantization_config=quantization_config,
                                                output_hidden_states=True,     
                                                device_map='auto')
    hm.register_hook(model, 0)

    text = "Once upon a time"
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_length=15, output_hidden_states=True)
    
    hm.remove_hooks()

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()