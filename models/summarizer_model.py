class SummarizerModel(BaseModel):
    def __init__(self, model_name, quantization_config, device):
        # Riutilizza l'inizializzazione di BaseModel
        super().__init__(model_name, quantization_config, device)
    
    def truncated(self, input_string, n_tokens):
        """Truncate the input string to a specific number of tokens."""
        
        try:
            tokens = self.tokenizer.encode(input_string)  # Usa self.tokenizer ereditato da BaseModel
            truncated_tokens = tokens[:n_tokens]
            truncated_string = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_string
        except Exception as e:
            print(f"Error truncating string: {e}")
            return input_string[:100]  # Fallback to simple string slicing
    
    def summarize(self, text, max_new_tokens=800):
        """Generate a summary of the provided text."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes content concisely."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ]
        
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.device)
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            counter = len(self.tokenizer.encode(summary))
            
            return summary, counter
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary", 0