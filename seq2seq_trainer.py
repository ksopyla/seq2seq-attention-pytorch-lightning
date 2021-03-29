import os
import sys
from argparse import ArgumentParser
import random



# # python.dataScience.notebookFileRoot=${fileDirname}
# wdir = os.path.abspath(os.getcwd() + "/../../")
# sys.path.append(wdir)

# print(sys.path)
# print(wdir)


import text_loaders as tl
import rnn_encoder_decoder as encdec

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plfunc
from pytorch_lightning.loggers import TensorBoardLogger



#%%
class Seq2SeqCorrector(pl.LightningModule):
    """Encoder decoder pytorch module for trainning seq2seq model with teacher forcing

    Module try to learn mapping from one sequence to antother. This implementation try to learn to reverse string of chars
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--emb_dim", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parser

    def __init__(
        self,
        vocab_size,
        padding_index=0,
        emb_dim=8,
        hidden_dim=32,
        dropout=0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size

        # dynamic, based on tokenizer vocab size defined in datamodule
        self.input_dim = vocab_size
        self.output_dim = vocab_size

        self.enc_emb_dim = emb_dim  # ENC_EMB_DIM
        self.dec_emb_dim = emb_dim  # DEC_EMB_DIM

        self.enc_hid_dim = hidden_dim  # ENC_HID_DIM
        self.dec_hid_dim = hidden_dim  # DEC_HID_DIM

        self.enc_dropout = dropout  # ENC_DROPOUT
        self.dec_dropout = dropout  # DEC_DROPOUT

        self.pad_idx = padding_index

        self.save_hyperparameters()

        self.max_epochs = kwargs["max_epochs"]

        self.learning_rate = 0.0005

        # self.input_src = torch.LongTensor(1).to(self.device)
        # self.input_src_len = torch.LongTensor(1).to(self.device)
        # self.input_trg = torch.LongTensor(1).to(self.device)

        # todo: remove it this blocks loading state_dict from checkpoints
        # Error(s) in loading state_dict for Seq2SeqCorrector:
        # size mismatch for input_src: copying a param with shape
        # torch.Size([201, 18]) from checkpoint,
        # the shape in current model is torch.Size([1]).
        # self.register_buffer("input_src", torch.LongTensor(1))
        # self.register_buffer("input_src_len", torch.LongTensor(1))
        # self.register_buffer("input_trg", torch.LongTensor(1))

        self._loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.attention = encdec.Attention(self.enc_hid_dim, self.dec_hid_dim)

        #    INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT
        self.encoder = encdec.Encoder(
            self.input_dim,
            self.enc_emb_dim,
            self.enc_hid_dim,
            self.dec_hid_dim,
            self.enc_dropout,
        )

        self.decoder = encdec.Decoder(
            self.output_dim,  # OUTPUT_DIM,
            self.dec_emb_dim,  # DEC_EMB_DIM,
            self.enc_hid_dim,  # ENC_HID_DIM,
            self.dec_hid_dim,  # DEC_HID_DIM,
            self.dec_dropout,  # DEC_DROPOUT,
            self.attention,
        )

        self._init_weights()

    def _init_weights(self):

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs TODO: change to registered buffer in pyLightning
        decoder_outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(
            self.device
        )

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        mask = self.create_mask(src)
        # mask = [batch size, src len]
        # without sos token at the beginning and eos token at the end

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        # starting with input=<sos> (trg[0]) token and try to predict next token trg[1] so loop starts from 1 range(1, trg_len)
        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            decoder_outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return decoder_outputs

    def loss(self, logits, target):

        return self._loss(logits, target)

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=5e-4)

        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = optim.LambdaLR(optimizer, ...)
        # return [optimizer], [scheduler]

        # optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = optim.lr_scheduler.InverseSquareRootLR(optimizer, self.lr_warmup_steps)
        # return (
        #     [optimizer],
        #     [
        #         {
        #             "scheduler": scheduler,
        #             "interval": "step",
        #             "frequency": 1,
        #             "reduce_on_plateau": False,
        #             "monitor": "val_loss",
        #         }
        #     ],
        # )
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=int(len(self.train_dataloader())),
                epochs=self.max_epochs,
                anneal_strategy="linear",
                final_div_factor=1000,
                pct_start=0.01,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        src_batch, trg_batch = batch

        src_seq = src_batch["src_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        src_seq = src_seq.transpose(0, 1)
        src_lengths = src_batch["src_lengths"]

        trg_seq = trg_batch["trg_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        trg_seq = trg_seq.transpose(0, 1)
        # trg_lengths = trg_batch["trg_lengths"]

        # resize input buffers, should speed up training and help
        # with memory leaks https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

        # self.input_src.resize_(src_seq.shape).copy_(src_seq)
        # self.input_src_len.resize_(src_lengths.shape).copy_(src_lengths)
        # self.input_trg.resize_(trg_seq.shape).copy_(trg_seq)

        # just for testing lr scheduler
        # output = torch.randn((*trg_seq.size(), self.output_dim), requires_grad=True, device=trg_seq.device)

        # output = self.forward(self.input_src, self.input_src_len, self.input_trg)
        # old version of forward, with tensors from dataloader
        output = self.forward(src_seq, src_lengths, trg_seq)

        # do not know if this is a problem, loss will be computed with sos token

        # without sos token at the beginning and eos token at the end
        output = output[1:].view(-1, self.output_dim)

        # trg = trg_seq[1:].view(-1)
        trg = trg_seq[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self.loss(output, trg)

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """validation is in eval mode so we do not have to use
        placeholder input tensors
        """

        src_batch, trg_batch = batch

        src_seq = src_batch["src_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        src_seq = src_seq.transpose(0, 1)
        src_lengths = src_batch["src_lengths"]

        trg_seq = trg_batch["trg_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        trg_seq = trg_seq.transpose(0, 1)
        trg_lengths = trg_batch["trg_lengths"]

        outputs = self.forward(src_seq, src_lengths, trg_seq, 0)

        # # without sos token at the beginning and eos token at the end
        logits = outputs[1:].view(-1, self.output_dim)

        # trg = trg_seq[1:].view(-1)

        trg = trg_seq[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self.loss(logits, trg)

        # take without first sos token, and reduce by 2 dimension, take index of max logits (make prediction)
        # seq_len * batch size * vocab_size -> seq_len * batch_size

        pred_seq = outputs[1:].argmax(2)

        # change layout: seq_len * batch_size -> batch_size * seq_len
        pred_seq = pred_seq.T

        # change layout: seq_len * batch_size -> batch_size * seq_len
        trg_batch = trg_seq[1:].T

        # compere list of predicted ids for all sequences in a batch to targets
        acc = plfunc.accuracy(pred_seq.reshape(-1), trg_batch.reshape(-1))

        # need to cast to list of predicted sequences (as list of token ids)   [ [seq1_tok1, seq1_tok2, ...seq1_tokN],..., [seqK_tok1, seqK_tok2, ...seqK_tokZ]]
        predicted_ids = pred_seq.tolist()

        # need to add additional dim to each target reference sequence in order to
        # convert to format needed by bleu_score function [ seq1=[ [reference1], [reference2] ], seq2=[ [reference1] ] ]
        target_ids = torch.unsqueeze(trg_batch, 1).tolist()

        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        bleu_score = plfunc.nlp.bleu_score(predicted_ids, target_ids, n_gram=3).to(
            self.device
        )  # torch.unsqueeze(trg_batchT,1).tolist())

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_bleu_idx",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss, acc, bleu_score





if __name__ == "__main__":


    # look to .vscode/launch.json file - there are set some args
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=int, default=32 * 10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/10k_sent_typos_wikipedia.jsonl",
    )
   
    # add model specific args
    parser = Seq2SeqCorrector.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    dm = tl.ABCSec2SeqDataModule(
        batch_size=args.batch_size,
        N_random_samples=args.N_samples,
        N_valid_size=args.N_valid_size,
        num_workers=args.num_workers,
    )

    # dm = tl.SeqPairJsonDataModule(
    #     path=args.dataset_path,
    #     batch_size=args.batch_size,
    #     n_samples=args.N_samples,
    #     n_valid_size=args.N_valid_size,
    #     num_workers=args.num_workers,
    # )

    dm.prepare_data()
    dm.setup("fit")

    # to see results run in console
    # tensorboard --logdir tb_logs/
    # then open browser http://localhost:6006/

    log_desc = f"RNN with attention model vocab_size={dm.vocab_size} data_size={dm.dims}, emb_dim={args.emb_dim} hidden_dim={args.hidden_dim}"
    print(log_desc)

    logger = TensorBoardLogger(
        "model_corrector", name="pl_tensorboard_logs", comment=log_desc
    )

    from pytorch_lightning.callbacks import LearningRateMonitor

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor]
    )  # , distributed_backend='ddp_cpu')

    model_args = vars(args)
    model = Seq2SeqCorrector(
        vocab_size=dm.vocab_size, padding_index=dm.padding_index, **model_args
    )

    # most basic trainer, uses good defaults (1 gpu)
    trainer.fit(model, dm)

# sample cmd

# python seq2seq_trainer.py --dataset_path /data/10k_sent_typos_wikipedia.jsonl \
# --gpus=2 --max_epoch=5 --batch_size=16 --num_workers=4 \
# --emb_dim=128 --hidden_dim=512 \
# --log_gpu_memory=True --weights_summary=full \
# --N_samples=1000000 --N_valid_size=10000 --distributed_backend=ddp --precision=16 --accumulate_grad_batches=4 --val_check_interval=640 --gradient_clip_val=2.0 --track_grad_norm=2

# tensorboard dev --logdir model_corrector/pl_tensorboard_logs/version??

