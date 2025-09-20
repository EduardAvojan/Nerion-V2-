class VectorStore:

    def __init__(self):
        self._mem = []

    def add(self, text: str):
        self._mem.append(text)

    def remove(self, text: str):
        self._mem = [m for m in self._mem if m != text]

    def search(self, query: str):
        return [m for m in self._mem if query.lower() in m.lower()]