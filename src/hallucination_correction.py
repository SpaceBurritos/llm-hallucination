from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from hooks.hook import HookManager
from word_importance import SentenceSimilarity
import torch
from scipy.stats import entropy


def main():
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                quantization_config=quantization_config,
                                                output_attentions=True, 
                                                attn_implementation="eager",
                                                device_map='auto')

    hm = HookManager(model, tokenizer)
    ss = SentenceSimilarity()

    hm.register_hooks([23], "early_exit")
    hm.register_hooks([16, 20, 24], "attn")

    text = "The capital of Brazil is "
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_length=15)
    hm.remove_hooks()

    generated_text =  tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    generated_tokens = []

    # for o in outputs[0]:
    #     generated_tokens.append(tokenizer.decode(o, skip_special_tokens=True))

    # generated_tokens = generated_tokens[len(text.split(" "))+1:]
    generated_tokens2 = outputs[0][len(text.split(" "))+1:]
    generated_tokens = outputs[0][:len(text.split(" "))+1]


    # if (len(generated_tokens2) == len(hm.early_exit_words[0])):
    for i in range(len(generated_tokens2)):
        print(f"Real: {generated_tokens2[i]}, Early: {hm.early_exit_words[i]}")
        temp = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        temp1 = tokenizer.decode(torch.cat((generated_tokens, torch.tensor([generated_tokens2[i]]).to('cuda'))), skip_special_tokens=True)
        temp2 = tokenizer.decode(torch.cat((generated_tokens, torch.tensor([hm.early_exit_words[i]]).to('cuda'))), skip_special_tokens=True)
        print(temp, "\n", temp1, "\n", temp2)
        rt = ss.compare(temp, temp1)
        et = ss.compare(temp, temp2)

        print(f"Real impact: {rt}, Early impact: {et}")

        generated_tokens = torch.cat((generated_tokens, torch.tensor([generated_tokens2[i]]).to('cuda')))

    # hm.early_exit_words

if __name__ == "__main__":
    main()