class Issue:
    def __init__(self, digest=None, body=None, labels=None):
        self.digest = digest
        self.body = body
        self.labels = labels

    def __repr__(self):
        return "Issue hash: {}, Issue body: {}".format(digest, body)
