# Exciton NLP - A tool for natural language processing

Exciton NLP is designed and maintained by ExcitonX for different NLP tasks, including multilingual classification, NER, translation, etc.

## Installation
Use ``pip`` to install exciton. Run:

```
pip install -U exciton
```

## Usage

```

from exciton.nlp.translation import M2M100

model = M2M100(model="m2m100_1.2b", device="cuda")
source = [
    {"id": 1, "source": "I love you!", "source_lang": "en", "target_lang": "zh"},
    {"id": 2, "source": "我爱你！", "source_lang": "zh", "target_lang": "en"}
]
results = model.predict(source)

```