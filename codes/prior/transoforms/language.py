import random

class SentenceShuffle():
    def __init__(self) -> None:
        pass

    def __call__(self, sentences):
        random.shuffle(sentences)
        new_sentences = ''
        for sentence in sentences:
            new_sentences += sentence
        return new_sentences
