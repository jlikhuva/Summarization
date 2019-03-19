import sys
import torch
import time
import numpy as np
from net import StyleTransfer
from decoder import STDecoder
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

args = {
	'--train-src': '../data/gigaword/train.article.txt',
	'--train-tgt': '../data/gigaword/train.title.txt',
	'--train-ext': '../data/gigaword/bottom_up_train.src.txt',
	'--dev-src' : '../data/gigaword/valid.article.filter.txt',
	'--dev-tgt':'../data/gigaword/valid.title.filter.txt',
	'--dev-ext': '../data/gigaword/bottom_up_valid.src.txt',
	'--vocab':'vocab.json',
	'--cuda': False,
	'--batch-size': 32,
	'--clip-grad': 5.0,
	'--valid-niter': 2000,
	'--log-every': 10,
	'--max-epoch': 30,
	'--uniform-init': 0.1,
	'--lr': 0.001,
}

def forward(source_document, extractive_summary,
        target, encoder, decoder, vocab, device=torch.device('cpu')):
    """

    @param source_document (List[List[str]]): List of the source document that needs to be summarized.
    @param extractive_summary (List[List[int]]): List of words produced by our sequence tagger
    @param target (List[List[str]]): List of words. They are the gold summaries.
    @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                log-likelihood of generating the gold-standard target sentence for
                                each example in the input batch. Here b = batch size.
    """
    source_lengths = [len(s) for s in source_document]
    source_padded = vocab.src.to_input_tensor(source_document, device=device)
    target_padded = vocab.tgt.to_input_tensor(target, device=device)
    extractive_padded = torch.tensor([[i for i in l] for l in source_padded])
    # print(extractive_padded.shape, source_padded.shape)
    for i, l in enumerate(extractive_summary):
    	for j, v in enumerate(l):
	    	if v == 0:
        		extractive_padded[j][i] = 0
    # print(extractive_padded[:, 0], source_padded[:, 0])
    latent_state = encoder(content=extractive_padded, style=source_padded)
    out = decoder(latent_state, latent_state, target_padded)
    return out

def train(args):
    """
    Trains the style transfer summarizer.
    Portions Derived from CS 224 A4 Code.
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    train_data_ext = read_corpus(args['--train-ext'], source='src')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    dev_data_ext = read_corpus(args['--dev-ext'], source='src')

    train_data = list(zip(train_data_src, train_data_tgt, train_data_ext))
    dev_data = list(zip(dev_data_src, dev_data_tgt, dev_data_ext))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    vocab = Vocab.load(args['--vocab'])

    encoder = StyleTransfer(vocab_size=len(vocab.tgt))
    decoder = STDecoder(vocab_size=len(vocab.tgt))
    encoder.train(); decoder.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in encoder.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
        for p in decoder.parameters():
        	p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    encoder = encoder.to(device); decoder=decoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(args['--lr']))
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents, ext_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(src_sents)
            example_losses = -forward(src_sents, ext_sents, tgt_sents, encoder, decoder, vocab) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1
                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)
def main():
    assert(torch.__version__ == "1.0.0")
    seed = 0
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    train(args)

if __name__ == '__main__':
    main()
