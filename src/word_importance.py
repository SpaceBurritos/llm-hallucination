from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example sentences
sentence1 = "There were two things that were important to Tracey."
# sentence1 = "A fast dark dog leaps over a sleepy bird."

sentence1 = sentence1.split()

# Compute embeddings
for s in range(1,len(sentence1)+1):
    small = " ".join(sentence1[:s-1])
    long = " ".join(sentence1[:s])
    print(small, long)
    embedding1 = model.encode(small, convert_to_tensor=True)
    embedding2 = model.encode(long, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)

    print(1 - similarity)