topics_pub = {}
topics_sub = {}


def spin():
    pass


def init_node(obj1, anonymous=True):
    pass


class Publisher:
    def __init__(self, topic, object_type, queue_size):
        self.topic = topic
        self.object_type = object_type
        self.queue_size = queue_size
        if topic not in topics_pub:
            topics_pub[topic] = []
        topics_pub[topic].append(self)

    def publish(self, obj):

        for sub in topics_sub[self.topic]:
            sub.callback(obj)

    def unregister(self):
        pass


class Subscriber:
    def __init__(self, topic, object_type, callback):
        self.topic = topic
        self.object_type = object_type
        self.callback = callback
        if topic not in topics_sub:
            topics_sub[topic] = []
        topics_sub[topic].append(self)
