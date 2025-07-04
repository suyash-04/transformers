import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader  , random_split
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import os
import warnings

from model import build_transformer
from dataset import BilingualDataset , casual_mask
from tqdm import tqdm
from config import get_weights_file_path , latest_weights_file_path , get_config

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import torchmetrics

from torch.utils.tensorboard import SummaryWriter


def greedy_decode( model , source ,source_mask, tokernizer_src, tokenizer_tgt, max_len , device):
    sos_idx =tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output  = model.encode(source , source_mask)
    decoder_input = torch.empty(1, 1, dtype=source.dtype).to(device).fill_(sos_idx)


    
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decode_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        out = model.decode(encoder_output, decoder_input ,decode_mask, source_mask)
        
        prob = model.projection(out[: ,-1])
        _, next_word = torch.max(prob , dim=1 )
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1,).type_as(source).fill_(next_word.item()).to(device)] , dim = 1
        )
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)
    
def run_validation(model , validation_ds , tokenizer_src , tokenizer_tgt , max_len , device , print_msg , global_step , writer , num_examples = 2):
    model.eval()
    count = 0 
    source_texts = []
    expected = []
    predicted =[]
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count +=1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) ==1, "Batch size must be one"
            
            model_out = greedy_decode(model , encoder_input , encoder_mask, tokenizer_src, tokenizer_tgt , max_len , device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().tolist(), skip_special_tokens=True)

            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    if writer:
        # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()   
                
            
            


def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]



def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.exists():
        # Initialize tokenizer with unknown token
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]", vocab={}))
        
        # Set pre-tokenizer
        whitespace_token = Whitespace()
        tokenizer.pre_tokenizer = whitespace_token
        # Set up trainer
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Save tokenizer
        tokenizer.save(str(tokenizer_path))

    # Load and return tokenizer
    return Tokenizer.from_file(str(tokenizer_path))


def get_ds(config):
    ds_all = load_dataset("sharad461/ne-en-parallel-208k", split='train')

    # lets use only 10% of the data
    subset_size = int(0.1 * len(ds_all))
    ds_raw = ds_all.select(range(subset_size))
    print(f"Using {subset_size} samples out of {len(ds_all)}")
    print(ds_raw[0])

    # build the tokenizers
    tokenizer_src = get_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config['lang_tgt'])

    # now 90% for training and remaning for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    tokenizer_src = get_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config['lang_tgt'])
    
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Initialize max length variables
    max_len_src = 0
    max_len_tgt = 0
    
    # Calculate max lengths
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max source length: {max_len_src}, Max target length: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len , vocab_tgt_len):
    model = build_transformer(vocab_src_len , vocab_tgt_len , config['seq_len'] , config['seq_len'] , config['d_model'])
    return model

def train_model(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard writer
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'] , eps= 1e-9 )
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_src.token_to_id('[PAD]'),
    label_smoothing=0.1
    ).to(device)

    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch : 02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output , decoder_input , decoder_mask , encoder_mask)
            proj_output = model.projection(decoder_output)
            
            label = batch['label'].to(device)
            
            loss = loss_fn(proj_output.view( - 1 , tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)    
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
            
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)