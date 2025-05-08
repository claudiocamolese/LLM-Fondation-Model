import torch
from model import BaseModel

class SentenceSimilarity(BaseModel):
    def __init__(self, model_name, quantization_config, device):
        super().__init__(model_name, quantization_config, device)

    # Funzione per ottenere gli embedding delle frasi
    def get_embeddings(self, sentences):
        # Tokenizza le frasi
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            # Calcola l'embedding usando l'output dell'ultima hidden layer
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings

    def calculate_cosine_similarity(self, source_sentence, sentences):
        # Ottieni gli embedding delle frasi
        source_embedding = self.get_embeddings([source_sentence])
        sentence_embeddings = self.get_embeddings(sentences)

        cosine_sim = torch.nn.functional.cosine_similarity(source_embedding, sentence_embeddings)

        return cosine_sim.numpy()  