#!/usr/bin/env python

'''
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import requests
import re
import os
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of keys/queries/values per head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            # Expand the mask to match the number of heads
            mask = mask.unsqueeze(1)  # Add a head dimension
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len):
        super(TransformerLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(input_ids.device)

        embedded = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(embedded)

        for block in self.transformer_blocks:
            x = block(x, mask)

        logits = self.fc(x)
        return logits

# Fetch most frequent English words
def get_most_frequent_words():
    """
    Fetches the most frequent English words from a comprehensive word list.

    Args:
        n (int): The number of most frequent words to retrieve.

    Returns:
        list: A list of the 'n' most frequent English words, or an empty list on error.
    """
    try:
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"  # A much larger word list
        #url = "https://gist.githubusercontent.com/deekayen/4148741/raw/98d35708fa344717d8eee15d11987de6c8e26d7d/1-1000.txt"

        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        words = response.text.splitlines()

        # Simple frequency approximation: assume alphabetical order roughly correlates with frequency
        return words

    except requests.exceptions.RequestException as e:
        print(f"Error fetching word list: {e}")
        return []
    except Exception as e: #catch other errors.
        print(f"An unexpected error occurred: {e}")
        return []

# Fetch sci-fi text from the web
def fetch_sci_fi_text(url="https://www.gutenberg.org/files/84/84-0.txt"): #h.g wells time machine
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        text = response.text
        # Clean the text (remove Gutenberg headers/footers, non-text content)
        start_index = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK THE TIME MACHINE ***")
        end_index = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK THE TIME MACHINE ***")
        if start_index != -1 and end_index != -1:
            text = text[start_index + len("*** START OF THIS PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"):end_index]
        text = re.sub(r'[^a-zA-Z\s.]', '', text) #remove non-alphanumeric and keep spaces and periods.
        text = re.sub(r'\s+', ' ', text).strip() #remove extra spaces.
        return text.lower()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sci-fi text: {e}")
        return ""

def train_sentencepiece_tokenizer(text, vocab_size=20234, model_prefix="sentencepiece_model", max_sentence_length=410367):
    with open("temp.txt", "w") as f:
        f.write(text)
    spm.SentencePieceTrainer.train(
        f'--input=temp.txt --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --max_sentence_length={max_sentence_length}'
    )
    os.remove("temp.txt")

def tokenize(text, sp):
    encoded = sp.encode(text, out_type=int)
    return encoded

def detokenize(tokens, sp):
    decoded = sp.decode(tokens)
    return decoded

def generate_training_data(text, max_length, sequence_length, sp):
    data = []
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence = "[START]" + sentence + ".[END]"
        tokens = tokenize(sentence, sp)
        if len(tokens) > max_length:
            continue
            #tokens = tokens[:max_length]
        for i in range(0, len(tokens) - sequence_length, sequence_length):
            data.append(tokens[i:i + sequence_length])
        if len(tokens) % sequence_length != 0:
            data.append(tokens[-(len(tokens) % sequence_length):])
    return data

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch, pad_token_id):
    max_len = max(len(seq) for seq in batch)
    padded_batch = [seq + [pad_token_id] * (max_len - len(seq)) for seq in batch]
    input_ids = torch.LongTensor(padded_batch)
    targets = input_ids[:, 1:].clone()
    targets = torch.cat([targets, torch.full((targets.shape[0], 1), pad_token_id)], dim=1)
    mask = torch.ones(input_ids.size(0), input_ids.size(1), input_ids.size(1)).tril()
    return input_ids, targets, mask

def collate_wrapper(batch, pad_token_id):
    return custom_collate(batch, pad_token_id)

def generate_sci_fi_story(model, start_text="[START]", max_length=150, temperature=0.7, device="cpu", sp=None):
    model.eval()
    input_ids = torch.tensor([tokenize(start_text, sp)]).to(device)
    generated_text = start_text
    end_count = 0
    with torch.no_grad():
        with tqdm(range(max_length), total=max_length, desc=f"Progress") as pbar:
            for _ in range(max_length):
                mask_gen = torch.ones(1, input_ids.size(1), input_ids.size(1)).tril().to(device)
                logits = model(input_ids, mask_gen)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if next_token != sp.piece_to_id("."):
                    generated_text += " " + sp.decode([next_token])
                else:
                    generated_text += sp.decode([next_token])
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                if next_token == sp.piece_to_id("[END]"):
                    end_count += 1
                    if end_count >= 5:
                        break
    return detokenize(tokenize(generated_text, sp), sp)

if __name__ == '__main__': # Wrap main code in this block
    sci_fi_text = fetch_sci_fi_text()
    train_sentencepiece_tokenizer(sci_fi_text)
    sp = spm.SentencePieceProcessor(model_file='sentencepiece_model.model')
    vocab_size = sp.get_piece_size()
    pad_token_id = sp.piece_to_id("[PAD]")

    # Example usage:
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 150
    batch_size = 8
    sequence_length = 100
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    training_data = generate_training_data(sci_fi_text, max_seq_len, sequence_length, sp)
    validation_split = 0.1
    validation_size = int(len(training_data) * validation_split)
    validation_data = training_data[:validation_size]
    training_data = training_data[validation_size:]

    # Pad data to the maximum length of both training and validation sets
    max_len = max(max(len(seq) for seq in training_data), max(len(seq) for seq in validation_data))

    for i in range(len(training_data)):
        pad_len = max_len - len(training_data[i])
        training_data[i].extend([pad_token_id] * pad_len)

    for i in range(len(validation_data)):
        pad_len = max_len - len(validation_data[i])
        validation_data[i].extend([pad_token_id] * pad_len)

    train_dataset = TextDataset(training_data)
    val_dataset = TextDataset(validation_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_wrapper, pad_token_id=pad_token_id), pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_wrapper, pad_token_id=pad_token_id), pin_memory=True, num_workers=4)

    model = TransformerLLM(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accumulation_steps = 4
        optimizer.zero_grad()

        # Wrap the iteration with tqdm
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (input_ids, targets, mask) in pbar:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                mask = mask.to(device)

                logits = model(input_ids, mask)
                loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / accumulation_steps

                loss.backward() # standard backward pass

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step() # standard optimizer step
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    pbar.set_postfix({"loss": total_loss / (i + 1)})

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # validation phase
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        with tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation") as pbar: # Wrap with tqdm
            for i, (input_ids, targets, mask) in pbar:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                mask = mask.to(device)

                logits = model(input_ids, mask)
                val_loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
                validation_loss += val_loss.item()
                pbar.set_postfix({"val_loss": validation_loss / (i + 1)}) #add validation loss to tqdm.

    print(f"Validation Loss: {validation_loss / len(val_loader)}")

    torch.save(model.state_dict(), "transformer_llm.pth")

    # inference generate story
    story = generate_sci_fi_story(model, device=device, sp=sp)
    print(story)
