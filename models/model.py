class BaseModel():
    
    def __init__(self, model_name, quantization_config, device):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        quantization_config = self.quantization_config,
        device_map = self.device,
        use_cache =  False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
        
    def forget_all(self):
        import gc
        import os
        import sys

        # Gestisci il modello in modo difensivo
        if self.model is not None:
            try:
                # Lista di attributi comuni di cache da ripulire
                cache_attrs = ["cache", "kv_cache", "past_key_values", "attention_mask",
                            "key_value_memory_cache", "memory", "buffer"]

                # Pulisci attributi senza usare dir()
                for attr in cache_attrs:
                    try:
                        if hasattr(self.model, attr):
                            setattr(self.model, attr, None)
                    except:
                        pass

                # Metodi comuni di pulizia cache
                clean_methods = ["clear_cache", "reset_cache", "empty_cache", "free_memory"]
                for method in clean_methods:
                    try:
                        if hasattr(self.model, method) and callable(getattr(self.model, method)):
                            getattr(self.model, method)()
                    except:
                        pass
            except:
                pass

            # Forza il modello a None
            self.model = None

        # Gestisci il tokenizer in modo difensivo
        if self.tokenizer is not None:
            try:
                # Attributi comuni di cache del tokenizer
                tokenizer_cache_attrs = ["cache", "encoder_cache", "decoder_cache"]
                for attr in tokenizer_cache_attrs:
                    try:
                        if hasattr(self.tokenizer, attr):
                            setattr(self.tokenizer, attr, None)
                    except:
                        pass
            except:
                pass

            # Forza il tokenizer a None
            self.tokenizer = None

        # Libera memoria CUDA
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except:
                    pass
        except:
            pass
        
    