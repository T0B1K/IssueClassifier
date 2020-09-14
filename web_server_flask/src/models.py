from hashlib import sha1


class Issue:
    def __init__(self, body: str):
        self.body = body
        self.labels = []
        self.digest = (sha1((self.body).encode('utf-8')).hexdigest())[:6]

    def __repr__(self):
        return "Issue hash: {}, issue body: {}, ".format(digest, body)
