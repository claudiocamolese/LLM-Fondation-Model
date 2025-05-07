from model import BaseModel

class SummarizerModel(BaseModel):
    
    def __init__(self, model_name, quantization_config, device):
        super().__init__(model_name, quantization_config, device)
      
        
    def summarize(self, prompt, max_new_tokens=300, do_sample=False):
        
        try:
            messages = [
                    {"role": "system", "content": "Provide only a concise summary of the user's message. Do NOT include any reasoning or inner thoughts. Do not add new informations. Output only the final summary text."},
                    {"role": "user", "content": prompt}
                ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                do_sample=do_sample
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens= max_new_tokens,
                streamer=streamer
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            counter = len(self.tokenizer.encode(response))


            return response, counter
        
        finally:
            pass
            # Clean up everything even if an error occurs
            # forget_all(model=self.model,tokenizer=self.tokenizer)
    def truncated(self, input_string, n_tokens=800):

        try:
          tokens = tokenizer.encode(input_string)
          truncated_tokens = tokens[:n_tokens]
          truncated_string = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
          return truncated_string
        finally:
            pass
        # Clean up tokenizer if it was created
        # if tokenizer:
        #    self.forget_all()