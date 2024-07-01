# from sentence_transformers import SentenceTransformer, util
from transformers import BertForMaskedLM, BertTokenizer
from scipy.special import softmax
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance
import torch 

import numpy as np

class SentenceSimilarity:

    def __init__(self):
        # self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def compare(self, sent1, sent2):
        # embedding1 = self.model.encode(sent1, convert_to_tensor=True)
        # embedding2 = self.model.encode(sent2, convert_to_tensor=True)
        # similarity = util.pytorch_cos_sim(embedding1, embedding2)
        probs_sent1 = self.get_probabilities(sent1)
        probs_sent2 = self.get_probabilities(sent2)
        p_distance = distance.jensenshannon(probs_sent1, probs_sent2)
        # print(similarity)
        return p_distance
    
    def get_probabilities(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = logits.mean(dim=1)
        return softmax(probs.squeeze())

if __name__ == "__main__":

    ss = SentenceSimilarity()
    sentence1 = "There were two things that were important to Tracey."
    # sentence1 = "A fast dark dog leaps over a sleepy bird."

    sentence1 = sentence1.split()

    for s in range(1,len(sentence1)+1):
        small = " ".join(sentence1[:s-1])
        long = " ".join(sentence1[:s])
        print(small, long)
        similarity = ss.compare(small, long)
        print(1 - similarity)