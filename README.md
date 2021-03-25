# Seq2Seq RNN model with attention as pytorch-lightning module


Code is modified version of @bentrevett great tutorial https://github.com/bentrevett/pytorch-seq2seq

Especially implements as Pytorch-lightning modules Encoder, Decoder and Seq2Seq trainer.

Implementation with Pytorch-Lightning allows:

- training in distributed environments (many GPUS)
- logging to Tensoboard
- customize DataModule to your specific use case (your data)
- remove dependency of TorchText


## How to run the code

Main file is [seq2seq_trainer.py](seq2seq_trainer.py) just run it in your IDE.


* Visual Studio Code users - there is launch.json in .vscode folder with settings and args


If you want to run in from terminal

```
python seq2seq_trainer.py --dataset_path /data/10k_sent_typos_wikipedia.jsonl \
 --gpus=2 --max_epoch=5 --batch_size=16 --num_workers=4 \
 --emb_dim=128 --hidden_dim=512 \
 --log_gpu_memory=True --weights_summary=full \
 --N_samples=1000000 --N_valid_size=10000 --distributed_backend=ddp --precision=16 --accumulate_grad_batches=4 --val_check_interval=640 --gradient_clip_val=2.0 --track_grad_norm=2
```

Ryn tensorboard

```
tensorboard dev --logdir model_corrector/pl_tensorboard_logs/version??

```

### Data modules

There are two data modules: 

* [ABCSec2SeqDataModule](text_loaders.py#L30) - generates artificial data
* [SeqPairJsonDataModule](text_loaders.py#L247) - read data from json line file

you should uncomment which one you want to use [seq2seq_trainer.py](seq2seq_trainer.py#L391).

### Tokenizers (token encoders)

Project use two tokenizers (token encoders) [CharTokenizerEncoder](text_loaders.py#L497) [BiGramTokenizerEncoder](text_loaders.py#L552) in each data module you can change it (find in code and uncomment or plug in yours).

