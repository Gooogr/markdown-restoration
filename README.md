# Markdown Restoration using Transformer Models

Based on official implementation of the paper [*Punctuation Restoration using Transformer Models for High-and Low-Resource Languages*](https://aclanthology.org/2020.wnut-1.18/).

## Data

Markdown datasets are provided in `data/markdown` directory. These are collected from [here](https://drive.google.com/file/d/0B13Cc1a7ebTuMElFWGlYcUlVZ0k/view).


## Model Architecture
Fine-tuned Transformer architecture based language model (e.g., BERT), followed by a bidirectional LSTM and linear layer that predicts target markdown token at each sequence position.

Supported markdown tags:
* Headers: H1, H2
* Separated text blocks
* Unordered lists


## Dependencies
Install PyTorch following instructions from [PyTorch website](https://pytorch.org/get-started/locally/). Remaining
dependencies can be installed with the following command
```bash
pip install -r requirements.txt
```

## Training
To train punctuation restoration model with optimal parameter settings:
```
python src/train.py --cuda=True --pretrained-model=roberta-large --freeze-bert=False --lstm-dim=-1 
--language=markdown --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out
```

#### Supported models for training
```
bert-base-uncased
bert-large-uncased
bert-base-multilingual-cased
bert-base-multilingual-uncased
xlm-mlm-en-2048
xlm-mlm-100-1280
roberta-base
roberta-large
distilbert-base-uncased
distilbert-base-multilingual-cased
xlm-roberta-base
xlm-roberta-large
albert-base-v1
albert-base-v2
albert-large-v2
```


## Pretrained Models (UPDATE THIS PART)
You can find pretrained models with augmentation here:
* [XLM-RoBERTa-large]()
* [XLM-RoBERTa-base]()


## Inference
You can run inference on unprocessed text file to produce punctuated text using `inference` module. 

Example script for English:
```bash
python src/inference.py --pretrained-model=roberta-large --weight-path=out/roberta-large-markdown.pt --language=en 
--in-file=data/test_markdown.txt --out-file=data/test_markdown_out.txt
```
This should create the text file with following output (UPDATE THIS PART):
```text
Tolkien drew on a wide array of influences including language, Christianity, mythology, including the Norse Völsunga saga, archaeology, especially at the Temple of Nodens, ancient and modern literature and personal experience. He was inspired primarily by his profession, philology. his work centred on the study of Old English literature, especially Beowulf, and he acknowledged its importance to his writings. 
```



## Test (UPDATE THIS PART)
Trained models can be tested on processed data using `test` module to prepare result.

For example, to test the best preforming English model run following command
```bash
python src/test.py --pretrained-model=roberta-large --lstm-dim=-1 --use-crf=False --data-path=data/test
--weight-path=weights/roberta-large-en.pt --sequence-length=256 --save-path=out
```
Please provide corresponding arguments for `pretrained-model`, `lstm-dim`, `use-crf` that were used during training the
model. This will run test for all data available in `data-path` directory.


## Citation

```
@inproceedings{alam-etal-2020-punctuation,
    title = "Punctuation Restoration using Transformer Models for High-and Low-Resource Languages",
    author = "Alam, Tanvirul  and
      Khan, Akib  and
      Alam, Firoj",
    booktitle = "Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wnut-1.18",
    pages = "132--142",
}
```
