import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention_Block(nn.Module):
    def __init__(self, ):
        super(Attention_Block, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, att_mask, pad_mask):
        # q, k, v shape: [batch_size, head, length, d_tensor]
        # pad_mask shape: [batch_size, length]
        length, tensor_dim = q.shape[2], q.shape[3]
        k_T = k.transpose(2, 3)
        att = (q.matmul(k_T)) / math.sqrt(tensor_dim)
        # add masks
        if att_mask is not None: att += att_mask
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, length, 1)
            att[pad_mask] = float('-inf')
        # calculate sofftmax
        att = self.softmax(att).matmul(v)
        return att

class MultiHeadAttention(nn.Module):
    def __init__(self, ndim, nhead):
        super(MultiHeadAttention, self).__init__()
        self.ndim, self.nhead = ndim, nhead
        self.w_q = nn.Linear(ndim, ndim)
        self.w_k = nn.Linear(ndim, ndim)
        self.w_v = nn.Linear(ndim, ndim)
        self.att = Attention_Block()
        self.w_out = nn.Linear(ndim, ndim)
    def forward(self, q, k, v, att_mask, pad_mask):
        batch_size, length = q.shape[0], q.shape[1]
        # phase 1: project in:
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # phase 2: split:
        split_dim = self.ndim // self.nhead
        q, k, v = (q.split(split_dim, dim=2),
                   k.split(split_dim, dim=2),
                   v.split(split_dim, dim=2))
        q, k, v = (torch.stack(q, dim=1),
                   torch.stack(q, dim=1),
                   torch.stack(q, dim=1))
        # phase 3: invoke attention module
        att_out = self.att(q, k, v, att_mask, pad_mask)
        # concat heads.
        att_out = att_out.permute(0,2,1,3).contiguous().view(batch_size, length, -1)
        # phase 4: project out:
        out = self.w_out(att_out)

        return out

class Encoder_Block(nn.Module):
    def __init__(self,
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        super(Encoder_Block, self).__init__()
        self.pre_norm = pre_norm
        self.self_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(ndim)
        self.ff1 = nn.Linear(ndim, ndim_feedforward)
        self.ff2 = nn.Linear(ndim_feedforward, ndim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.norm2 = nn.LayerNorm(ndim)
    def forward(self, x, attn_mask, padding_mask):
        # phase 1: self-attention
        y = self.norm1(x) if self.pre_norm else x
        x = x + self.self_attention(y,y,y,attn_mask=attn_mask,key_padding_mask=padding_mask)[0]
        if not self.pre_norm: x = self.norm1(x)
        # phase 2: feed forward
        y = self.norm2(x) if self.pre_norm else x
        x = x + self.dropout2(self.ff2(self.dropout1(F.relu(self.ff1(y)))))
        if not self.pre_norm: x = self.norm2(x)
        return x

class Decoder_Block(nn.Module):
    def __init__(self,
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        super(Decoder_Block, self).__init__()
        self.pre_norm = pre_norm
        self.self_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(ndim)
        self.cross_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(ndim)
        self.ff1 = nn.Linear(ndim, ndim_feedforward)
        self.ff2 = nn.Linear(ndim_feedforward, ndim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.norm3 = nn.LayerNorm(ndim)
    def forward(self, x, encoder_out, attn_mask, padding_mask):
        # phase 1: self-attention
        y = self.norm1(x) if self.pre_norm else x
        x = x + self.self_attention(y,y,y,attn_mask=attn_mask,key_padding_mask=padding_mask)[0]
        if not self.pre_norm: x = self.norm1(x)
        # phase 2: cross-attention
        y = self.norm2(x) if self.pre_norm else x
        x = x + self.cross_attention(y, encoder_out, encoder_out, attn_mask=attn_mask, key_padding_mask=padding_mask)[0]
        if not self.pre_norm: x = self.norm2(x)
        # phase 3: feed forward
        y = self.norm3(x) if self.pre_norm else x
        x = x + self.dropout2(self.ff2(self.dropout1(F.relu(self.ff1(y)))))
        if not self.pre_norm: x = self.norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 mode='all',
                 nlayer=[12,12],
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=True):
        super(Transformer, self).__init__()
        # sanity check
        if mode not in ['all', 'encoder_only', 'decoder_only']: raise ValueError("Unsupported mode.")
        if ndim % nhead != 0: raise ValueError("ndim must be divisible by nhead")

        self.vocabulary_size, self.mode, self.nhead, self.ndim, self.ndim_feedforward = vocabulary_size, mode, nhead, ndim, ndim_feedforward
        if self.mode == 'all': self.nlayer_encoder, self.nlayer_decoder = nlayer[0], nlayer[1]
        elif self.mode == 'encoder_only': self.nlayer_encoder, self.nlayer_decoder = nlayer[0], None
        else: self.nlayer_encoder, self.nlayer_decoder = None, nlayer[1]
        self.pre_norm = pre_norm

        if self.mode == 'all':
            self.encoder_layers = nn.ModuleList([Encoder_Block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               drop_out,
                                                               pre_norm) for _ in range(self.nlayer_encoder)])
            self.decoder_layers = nn.ModuleList([Decoder_Block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               drop_out,
                                                               pre_norm) for _ in range(self.nlayer_decoder)])
        elif self.mode == 'encoder_only':
            self.encoder_layers = nn.ModuleList([Encoder_Block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               drop_out,
                                                               pre_norm) for _ in range(self.nlayer_encoder)])
        else:
            self.decoder_layers = nn.ModuleList([Decoder_Block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               drop_out,
                                                               pre_norm) for _ in range(self.nlayer_decoder)])

        if self.pre_norm: self.final_norm = nn.LayerNorm(self.ndim)
        self.out = nn.Linear(self.ndim, self.vocabulary_size)
        # initialize parameters
        self._ini_para()

    def _ini_para(self):
        # for name, module in self.named_modules():
        #     print(f'{name}: {module.__class__.__name__}')
        print('transformer initializing...')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.normal_(m.in_proj_weight, mean=0, std=0.02)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.normal_(m.out_proj.weight, mean=0, std=0.02)
                nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self,
                encoder_in,
                encoder_out,
                decoder_in,
                encoder_attn_mask=None,
                decoder_attn_mask=None,
                encoder_padding_mask=None,
                decoder_padding_mask=None):
        """input data shape: [batch, length, embedding]"""
        if self.mode == 'all':
            encoder_out, decoder_out = encoder_in, decoder_in
            for layer in self.encoder_layers:
                encoder_out = layer(encoder_out, encoder_attn_mask, encoder_padding_mask)
            for layer in self.decoder_layers:
                decoder_out = layer(decoder_out, encoder_out, decoder_attn_mask, decoder_padding_mask)
            if self.pre_norm: decoder_out = self.final_norm(decoder_out)
            return self.out(decoder_out)
        elif self.mode == 'encoder_only':
            encoder_out = encoder_in
            for layer in self.encoder_layers:
                encoder_out = layer(encoder_out, encoder_attn_mask, encoder_padding_mask)
            if self.pre_norm: encoder_out = self.final_norm(encoder_out)
            return self.out(encoder_out)
        else:
            decoder_out = decoder_in
            for layer in self.decoder_layers:
                decoder_out = layer(decoder_out, encoder_out, decoder_attn_mask, decoder_padding_mask)
            if self.pre_norm: decoder_out = self.final_norm(decoder_out)
            return self.out(decoder_out)
        
if __name__ == '__main__':
    mha = MultiHeadAttention(35, 5)
    q, k, v = torch.randn(2, 3, 35), torch.randn(2, 3, 35), torch.randn(2, 3, 35)
    mha(q,k,v, None, None)
    from torchview import draw_graph

    # Perform a forward pass through the model
    model = Transformer(50000, 'all', [2, 2], 2, 512, 2048, 0.1, True)
    dummy_input = torch.randn(64, 200, 512)
    out = model(dummy_input, None, dummy_input)

    # Visualize the model
    model_visual = draw_graph(model, input_data=(dummy_input, None, dummy_input))

    # To display the visualization (in a Jupyter notebook for example)
    model_visual.visual_graph.render('model_graph', format='png')

