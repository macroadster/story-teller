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
import sentencepiece as spm
from tqdm import tqdm
import os

class MultiHeadAttention(nn.Module):
    # ... (MultiHeadAttention class definition - same as in your training script) ...
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
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
    # ... (PositionWiseFeedForward class definition - same as in your training script) ...
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    # ... (TransformerBlock class definition - same as in your training script) ...
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
    # ... (TransformerLLM class definition - same as in your training script) ...
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

def tokenize(text, sp):
    encoded = sp.encode(text, out_type=int)
    return encoded

def detokenize(tokens, sp):
    decoded = sp.decode(tokens)
    return decoded

def generate_sci_fi_story(model, start_text="[START]", max_length=500, temperature=0.8, device="cpu", sp=None):
    model.eval()
    input_ids = torch.tensor([tokenize(start_text, sp)]).to(device)
    generated_text = start_text
    end_count = 0
    with torch.no_grad():
        with tqdm(range(max_length), total=max_length, desc=f"Progress") as pbar:
            for _ in pbar:
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

def load_model_and_generate(model_path="transformer_llm.pth", sentencepiece_model_path="sentencepiece_model.model", device="cpu"):
    sp = spm.SentencePieceProcessor(model_file=sentencepiece_model_path)
    vocab_size = sp.get_piece_size()

    # Load model parameters (adjust as needed)
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 150

    model = TransformerLLM(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Generate stories
    story = generate_sci_fi_story(model, device=device, sp=sp)
    print(story)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    load_model_and_generate(device=device)
