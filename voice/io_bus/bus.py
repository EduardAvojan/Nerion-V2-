class EventBus:

    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, fn):
        self.subscribers.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        for fn in self.subscribers.get(topic, []):
            fn(payload)