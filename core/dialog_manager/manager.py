class DialogManager:

    def __init__(self, always_speak: bool=True):
        self.always_speak = always_speak

    def policy(self, intent: str) -> str:
        return 'speak' if self.always_speak else 'silent'