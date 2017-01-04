
"""
"""


from __future__ import print_function, division


class DataAlphabet(object):
    """
    """
    def __init__(self):
        self.tokens = list("abcdefghijklmnopqrstuvwxyz")

    def merge(self):
        pass

    def split(self, ptrain=0.8, pvalidate=0.1, ptest=0.1):
        pass

    def text(self, source, nchars=None):
        return self.tokens

    def sentences(self, source, nchars=None):
        s = self.text(source, nchars)
        s = s.replace('\r\n',' ')
        s = s.replace('\n',' ')
        sentences = tokenize.sent_tokenize(s)
        return sentences

    def tokens(self, source, nchars=None):
        sentences = self.sentences(source, nchars)
        tokens = []
        for sentence in sentences:
            # sentence = sentence.lower() # reduces vocab space
            words = tokenize.word_tokenize(sentence)
            tokens.extend(words)
            tokens.append('END') # add an END token to every sentence
        return tokens


if __name__=='__main__':

    s = """The dog barked. The cat meowed. The dog ran. The dog caught a frisbee. The cat
yawned. The cat slept. The dog barked at the cat. The cat woke up. The dog ran
away. The cat chased the dog. The dog chased the cat. The cat ran up a tree."""
    data = DataString(s)
    data.merge()
    data.split()
    s = data.tokens('train')
    print(s)
    s = data.tokens('test')
    print(s)


