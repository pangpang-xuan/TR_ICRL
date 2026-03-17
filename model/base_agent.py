class LLMAgent:
    def __init__(self, model_name, temperature=0.0, top_p=1.0) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        pass

    def get_response(self):
        pass
