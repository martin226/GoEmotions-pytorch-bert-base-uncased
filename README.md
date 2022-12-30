# GoEmotions Pytorch (bert-base-uncased fork)

Pytorch Implementation of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) with [Huggingface Transformers](https://github.com/huggingface/transformers)

## What is GoEmotions

Dataset labeled **58000 Reddit comments** with **28 emotions**

- admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise + neutral

## Training Details

- Use `bert-base-uncased`
- In paper, **3 Taxonomies** were used. I've also made the data with new taxonomy labels for `hierarchical grouping` and `ekman`.

  1. **Original GoEmotions** (27 emotions + neutral)
  2. **Hierarchical Grouping** (positive, negative, ambiguous + neutral)
  3. **Ekman** (anger, disgust, fear, joy, sadness, surprise + neutral)

### Vocabulary

- I've replaced `[unused0]`, `[unused1]` to `[NAME]`, `[RELIGION]` in the vocab, respectively.

```text
[PAD]
[NAME]
[RELIGION]
[unused3]
[unused4]
...
```

- I've also set `special_tokens_map.json` as below, so the tokenizer won't split the `[NAME]` or `[RELIGION]` into its word pieces.

```json
{
  "unk_token": "[UNK]",
  "sep_token": "[SEP]",
  "pad_token": "[PAD]",
  "cls_token": "[CLS]",
  "mask_token": "[MASK]",
  "additional_special_tokens": ["[NAME]", "[RELIGION]"]
}
```

### Requirements

- torch==1.4.0
- transformers==2.11.0
- attrdict==2.0.1

### Hyperparameters

You can change the parameters from the json files in `config` directory.

| Parameter         |      |
| ----------------- | ---: |
| Learning rate     | 5e-5 |
| Warmup proportion |  0.1 |
| Epochs            |   10 |
| Max Seq Length    |   50 |
| Batch size        |   16 |

## How to Run

For taxonomy, choose `original`, `group` or `ekman`

```bash
$ python3 run_goemotions.py --taxonomy {$TAXONOMY}

$ python3 run_goemotions.py --taxonomy original
$ python3 run_goemotions.py --taxonomy group
$ python3 run_goemotions.py --taxonomy ekman
```

## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions Github](https://github.com/google-research/google-research/tree/master/goemotions)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
