import torch
import torch.nn as nn
from Transformer import Transformer
import utils

class GPT(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_size,
                 training_seq_len,
                 nlayer=[12,12],
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=True):
        super(GPT, self).__init__()
        self.training_seq_len = training_seq_len
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # decoder_only transformer
        self.model = Transformer(vocabulary_size=vocabulary_size,
                                 mode='encoder_only',
                                 nlayer=nlayer,
                                 nhead=nhead,
                                 ndim=ndim,
                                 ndim_feedforward=ndim_feedforward,
                                 drop_out=drop_out,
                                 pre_norm=pre_norm)
        # cache attention mask and positional encoding.
        self.register_buffer('attn_mask', utils.gen_attn_mask(self.training_seq_len))
        self.register_buffer('pos_encoding', utils.gen_pos_encoding(10000, ndim))
        # initialize parameters
        self._ini_para()

    def _ini_para(self):
        for name, module in self.named_modules():
            print(f'{name}: {module.__class__.__name__}')
        print('embedding initializing...')
        for m in self.modules():
            if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0, std=0.02)
        
    def forward(self, seq, need_padding, padding_token):
        """seq shape: [batch, seq_length]"""
        assert seq.ndim == 2 and seq.shape[1] == self.training_seq_len
        batch_size, seq_len = seq.shape[0], seq.shape[1]
        # embedding & positional encoding.
        seq_embedding = self.embedding(seq)
        seq_embedding = seq_embedding + self.pos_encoding[0:seq_len,:]
        # feed into model.
        out = self.model(encoder_in=seq_embedding,
                         encoder_out=None,
                         decoder_in=None,
                         encoder_attn_mask=self.attn_mask,
                         decoder_attn_mask=None,
                         encoder_padding_mask=utils.gen_padding_mask(seq, padding_token) if need_padding else None,
                         decoder_padding_mask=None)
        
        return out

# test
if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_config = config['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt = GPT(model_config['vocabulary_size'],
              model_config['embedding_size'],
              model_config['training_seq_len'],
              model_config['nlayer'],
              model_config['nhead'],
              model_config['ndim'],
              model_config['ndim_feedforward'],
              model_config['drop_out'],
              model_config['pre_norm']).to(device).train()
    dummy_input = torch.randint(0, 50000, (64, 128)).to(device)
    print(gpt(dummy_input, model_config['need_padding'], model_config['padding_token']).shape)