import os
import numpy
import re


def OpenFile(file):
    path = os.path.join(os.path.dirname(__file__), file)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def Parameters(tdim, words):
    print(f"True Dimensions: {tdim}")
    print(f"Vocab: {len(words)}")


def Tokenize(text, by=None):
    return text.split(by)


def Probs(words):
    probs = {}

    for word in words:
        probs[word] = probs.get(word, 0) + 1

    return probs


class Model:
    def __init__(self, dim, batch, ddiv=32, mdiv=32):
        self.dim = dim
        self.batch = batch
        self.ddiv = ddiv
        self.mdiv = mdiv

        self.InitModel()

    def InitModel(self):
        if self.batch == self.dim or self.batch == self.mdiv:
            raise ValueError("Model size invalid")

        self.mdiv = int(self.mdiv * self.dim / self.batch)

        self.tdiv = self.ddiv * self.mdiv
        self.tdim = round(self.dim / self.ddiv)

    def Generate(self, words, input):
        output = []
        next_word = input

        for _ in range(self.mdiv + 1):
            pos_probs = {}

            for position, word in enumerate(words[:-1]):
                if word == next_word:
                    following = words[position + 1]
                    pos_probs[following] = pos_probs.get(following, 0) + 1

            if not pos_probs:
                break

            keys = list(pos_probs.keys())

            p = numpy.array(
                list(pos_probs.values()),
                dtype=numpy.float64
            )

            p = p ** (self.tdim / 32)
            p /= p.sum()

            next_word = str(
                numpy.random.choice(keys, p=p)
            )

            output.append(next_word)

        output_string = " ".join(output)

        output_string = re.sub(
            r"User:|Assistant:|[\[\]\'\",]",
            "",
            output_string
        )

        return re.sub(r"\s+", " ", output_string).strip()