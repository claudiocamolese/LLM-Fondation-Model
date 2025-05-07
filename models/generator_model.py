from model import BaseModel

class GeneratorModel(BaseModel):
    
    def __init__(self, model_name, quantization_config, device):
        super().__init__(model_name, quantization_config, device)
        
        
    def first_generate(self, prompt):
        
        start = t.time()

        messages = [
            {"role": "system", "content": "You are an assistant that solves problems step-by-step. Once you reach an answer, you MUST NOT verify or double-check it. Do not evaluate its correctness, just stop."},
            {"role": "user", "content": prompt}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                do_sample=do_sample
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)

            # Generate text
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )

            # Process the generated tokens to get the new part only
            processed_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(processed_ids, skip_special_tokens=True)[0]
            counter = len(self.tokenizer.encode(response))

            perplexity = self.calculate_perplexity(self.model, self.tokenizer, response, self.device)
            end = t.time()

            return response, counter, perplexity, end-start

        finally:
            # Only attempt cleanup if model and tokenizer were initialized
            if self.model is not None and self.tokenizer is not None:
                self.forget_all()
                
    def calculate_perplexity(self, text):
        
        """Calculate the perplexity of the generated text."""
        
        try:
            # Encode the text
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Create a labels tensor that's a copy of the input_ids
            labels = encodings.input_ids.clone()

            # Forward pass with labels for loss calculation
            with torch.no_grad():
                outputs = self.model(**encodings, labels=labels)

            # Get the loss
            neg_log_likelihood = outputs.loss.item()

            # Calculate perplexity: exp(loss)
            # Loss is already the mean negative log-likelihood
            perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

            return perplexity
        
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('nan')
        
    def n_generate(self, prompt, summary,  max_new_tokens=800, do_sample=False):

        start=t.time()

        messages = [
                {"role": "system", "content": (
        "You are a summarization assistant. "
        "Your task is to output only the final summary of the user’s message, "
        "in one concise sentence. "
        "Do NOT include any explanation, inner thoughts, or <think> tags. "
        "Do NOT describe what you're doing. Just return the summary directly, nothing else."
    )},
                {"role": "user", "content": "prompt: " + prompt + "summary: " + summary}
            ]
        
        try:
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                do_sample=do_sample
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            counter = len(self.tokenizer.encode(response))
            perplexity =self.calculate_perplexity(model,self.tokenizer,response,device)
            end=t.time()

            return end-start, response, counter, perplexity
        finally:
            # Clean up everything even if an error occurs
            self.forget_all()