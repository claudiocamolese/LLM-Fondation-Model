class SentenceSimilarity:
    def __init__(self, model_name,  quantization_config, device):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, sentence):
        # Tokenizza e calcola gli embedding
        inputs = self.tokenizer(sentence, padding=True, truncation= False, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Media dei token
        return embeddings

    def compute_similarity(self, sentence1, sentence2):
        # Funzione che accetta esattamente due frasi
        emb1 = self.get_embeddings([sentence1])
        emb2 = self.get_embeddings([sentence2])
        cosine_sim = F.cosine_similarity(emb1, emb2)
        return np.float16(cosine_sim.item())  # Restituisce un float

