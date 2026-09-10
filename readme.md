# Sonic

Sonic is a lightweight statistical AI model pipeline written in Python. Any of its functions can be incorporated in many projects.

## Installation

```bash
pip install sonic
```

## Usage

```python
import sonic

text = sonic.OpenFile("all.txt")
words = sonic.Tokenize(text)

model = sonic.Model(32, 16)
model.InitModel()

print(model.Generate(words, "the"))
```

## Features

* Simple text tokenization
* Word frequency analysis
* Statistical text generation
* Lightweight NumPy-based implementation

## Requirements
* NumPy

## License

MIT

## Incorporation
Sonic can be used as a pipeline in text-generation projects, tokenizing, or NLP experimenting for beginners.