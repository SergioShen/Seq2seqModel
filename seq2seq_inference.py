#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 00:14 2020/12/25
# @Author: Sijie Shen
# @File: seq2seq_inference
# @Project: Seq2seqModel

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 01:38 2020/8/11
# @Author: Sijie Shen
# @File: seq2seq_train
# @Project: Seq2seqModel


import torch
import torch.nn as nn
import torchtext
import random
import os
import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from utils.logger import get_logger
from utils.trainer import Trainer
from models.seq2seq import Seq2seqModel


def sub_tokens_to_tokens(sub_tokens):
    result = list()
    current = ''
    for sub_token in sub_tokens:
        if sub_token.endswith('$$'):
            current += sub_token[:-2]
        else:
            current += sub_token
            result.append(current)
            current = ''
    if len(current) != 0:
        result.append(current)

    return result


class MyTrainer(Trainer):
    def __init__(self, model, optimizer, lr_scheduler, loss_function, logger, writer, train_params):
        super().__init__(model, optimizer, lr_scheduler, loss_function, logger, writer, train_params)

    def train_batch(self, batch_data):
        pass

    def evaluate_batch(self, batch_data):
        pass

    def inference(self, dataset, name, src_vocab, tgt_vocab, max_decode_length=64):
        self.model.eval()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        output_file = (self.output_dir / ('%s_inference_result.json' % name)).open('w', encoding='utf-8')

        total = 0
        correct = 0
        refs = list()
        hyps = list()
        with torch.no_grad():
            for i, example in tqdm(enumerate(dataset)):
                total += 1
                input_seq = ['<sos>'] + example[self.train_params['dataset']['input_key']] + ['<eos>']
                input_length = torch.tensor([len(input_seq)], dtype=torch.int64).to(device)
                input_ids = [src_vocab.stoi[token] for token in input_seq]
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(1).to(device)

                model_output = self.model.inference(input_ids, input_length, max_decode_length)
                output_seq, output_length = model_output

                output_tokens = list()
                for idx in output_seq:
                    output_tokens.append(tgt_vocab.itos[idx])
                output_tokens = sub_tokens_to_tokens(output_tokens)

                refs.append([example['tgt_tokens']])
                hyps.append(output_tokens)
                if output_tokens == example['tgt_tokens']:
                    correct += 1
                example['hyp'] = output_tokens
                print(json.dumps(example), file=output_file)
        self.logger.info('%s accuracy: %.6f' % (name.capitalize(), correct / total))
        self.logger.info('%s BLEU: %.6f' % (name.capitalize(), corpus_bleu(refs, hyps) * 100))
        output_file.close()


def main(args):
    # Read params configuration
    params = json.load(open(args.params))
    model_params = params['model_params']
    train_params = params['train_params']
    output_dir = Path(train_params['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir()

    # Set up logger and TensorBoard writer
    logger = get_logger(output_dir / 'inference.log')
    logger.debug('PID: %d', os.getpid())
    logger.info('Using params file: %s' % args.params)
    logger.info(json.dumps(params))
    writer = None
    # Set random seed
    seed = 1911
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
    logger.debug('Set random seed to %d', seed)

    # Load vocabulary and dataset
    logger.debug('Loading vocabulary...')
    src_field = torchtext.data.field.Field(init_token='<sos>', eos_token='<eos>', include_lengths=True)
    src_field.vocab = pickle.load(Path(train_params['dataset']['src_vocab_path']).open('rb'))
    assert len(src_field.vocab) == model_params['src_vocab_size']
    tgt_field = torchtext.data.field.Field(init_token='<sos>', eos_token='<eos>', is_target=True)
    tgt_field.vocab = pickle.load(Path(train_params['dataset']['tgt_vocab_path']).open('rb'))
    assert len(tgt_field.vocab) == model_params['tgt_vocab_size']
    logger.info('Vocab loaded, src vocab size: %d, tgt vocab size: %d' % (len(src_field.vocab), len(tgt_field.vocab)))
    logger.debug('Loading dataset...')

    datasets = dict()
    dataset_names = ['valid', 'test']
    for name in dataset_names:
        dataset = [json.loads(line.strip()) for line in Path(train_params['dataset'][name]).open(encoding='utf-8')]
        datasets[name] = dataset
        logger.debug('%s size: %d' % (name.capitalize(), len(dataset)))

    # Build model
    logger.debug('Building model...')
    model = Seq2seqModel(model_params)
    optimizer = None
    lr_scheduler = None
    loss_function = getattr(nn, train_params['loss_function'])(**train_params['loss_function_args'])
    if torch.cuda.is_available():
        model.cuda()
    logger.debug('Model built')

    # Train model
    trainer = MyTrainer(model, optimizer, lr_scheduler, loss_function, logger, writer, train_params)
    logger.info('Loading model from %s', args.load)
    trainer.load_model(args.load)
    logger.info('Model loaded')

    logger.info('Inference begins...')
    logger.info('Evaluating %s' % train_params['dataset']['valid'])
    trainer.inference(datasets['valid'], 'valid', src_field.vocab, tgt_field.vocab, max_decode_length=2048)
    logger.info('Evaluating %s' % train_params['dataset']['test'])
    trainer.inference(datasets['test'], 'test', src_field.vocab, tgt_field.vocab, max_decode_length=2048)


if __name__ == '__main__':
    parser = ArgumentParser('Tree positional encoding experiment main function.')
    parser.add_argument('-p', '--params', action='store',
                        help='Path of configuration file, should be a .json file')
    parser.add_argument('-l', '--load', action='store', default=None,
                        help='Load a model from given path')
    args = parser.parse_args()

    if not args.load:
        print('Should specify a checkpoint for inference')
        exit(-1)

    main(args)
