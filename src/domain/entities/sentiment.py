class Sentiment:
    def __init__(
        self,
        text: str
    ):
        self.text = text
        
        self._validate()
        
    def _validate(self):
        
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty") 
        