import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

"""################################################
    Linear Regression Model
################################################"""
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.output = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.output(x)) # Use relu because photon count always >= 0
        
        return out
    
"""################################################ 
    MLP (Multi-Layer Perceptron)
################################################"""
class DNN4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN4, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.hidden4 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.25)
        self.output = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        # x = self.dropout(x) # random dropout on first 3 layers
        x = F.relu(self.hidden2(x))
        # x = self.dropout(x) # random dropout on first 3 layers
        x = F.relu(self.hidden3(x))
        # x = self.dropout(x) # random dropout on first 3 layers
        x = F.relu(self.hidden4(x))
        out = F.relu(self.output(x)) # Use relu because photon count always >= 0
        
        return out
    
"""################################################ 
    ResNet (Residual Network)
        -> deeper than MLP
################################################"""
class ResNet4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet4, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.resblock1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), 
                                       nn.Linear(128, 128),)
        self.resblock2 = nn.Sequential(nn.Linear(128+input_dim, 128), nn.ReLU(), 
                                       nn.Linear(128, 128))
        self.resblock3 = nn.Sequential(nn.Linear(128+input_dim, 128), nn.ReLU(), 
                                       nn.Linear(128, 128))
        self.resblock4 = nn.Sequential(nn.Linear(128+input_dim, 128), nn.ReLU(), 
                                       nn.Linear(128, 128))
        self.output = nn.Sequential(nn.Linear(128+input_dim, output_dim), nn.ReLU())
        
    def forward(self, x):
        h0 = self.hidden1(x)
        h1 = self.resblock1(h0)
        h2 = self.resblock2(torch.concat([h1+h0, x], axis=-1))
        h3 = self.resblock3(torch.concat([h2+h1, x], axis=-1))
        h4 = self.resblock4(torch.concat([h3+h2, x], axis=-1))
        out = self.output(torch.concat([h4+h3, x], axis=-1)) # Use relu because photon count always >= 0
        # out = out + 1e-3 # add a constant amount of count for smoother spectra (based on graphs in 3c50 paper)
        
        return out
    
"""######################################################
    ResNet (Residual Network)
        -> same-size ResNet
        -> multiple branches for diff energy ranges
        -> simulate MTL
######################################################"""
class MTL_ResNet4(nn.Module):
    def __init__(self, input_dim, soft_dim, medium_dim, hard_dim):
        super(MTL_ResNet4, self).__init__()
        
        """ Shared Layers"""
        self.hidden1 = nn.Linear(input_dim, 128)
        self.resblock1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), 
                                       nn.Linear(128, 128),)
        self.resblock2 = nn.Sequential(nn.Linear(128+input_dim, 128), nn.ReLU(), 
                                       nn.Linear(128, 128))
        self.resblock3 = nn.Sequential(nn.Linear(128+input_dim, 128), nn.ReLU(), 
                                       nn.Linear(128, 128))
        
        """ Task-specific layers: Soft photon counts"""
        self.resblock_soft = nn.Sequential(nn.Linear(128+input_dim, 32), nn.ReLU(), 
                                           nn.Linear(32, 32))
        self.output_soft = nn.Sequential(nn.Linear(32+input_dim, soft_dim), nn.ReLU())
        """ Task-specific layers: Medium photon counts"""
        self.resblock_medium = nn.Sequential(nn.Linear(128+input_dim, 32), nn.ReLU(), 
                                             nn.Linear(32, 32))
        self.output_medium = nn.Sequential(nn.Linear(32+input_dim, medium_dim), nn.ReLU())
        """ Task-specific layers: Hard photon counts"""
        self.resblock_hard = nn.Sequential(nn.Linear(128+input_dim, 32), nn.ReLU(), 
                                           nn.Linear(32, 32))
        self.output_hard = nn.Sequential(nn.Linear(32+input_dim, hard_dim), nn.ReLU())
        
    def forward(self, x):
        """ Shared layers"""
        h0 = self.hidden1(x)
        h1 = self.resblock1(h0)
        h2 = self.resblock2(torch.concat([h1+h0, x], axis=-1))
        h3 = self.resblock3(torch.concat([h2+h1, x], axis=-1))
        
        """ Task-specific layers"""
        h_soft = self.resblock_soft(torch.concat([h3+h2, x], axis=-1))
        h_medium = self.resblock_medium(torch.concat([h3+h2, x], axis=-1))
        h_hard = self.resblock_hard(torch.concat([h3+h2, x], axis=-1))
        
        out_soft = self.output_soft(torch.concat([h_soft, x], axis=-1)) # Use relu because photon count always >= 0
        out_medium = self.output_medium(torch.concat([h_medium, x], axis=-1)) # Use relu because photon count always >= 0
        out_hard = self.output_hard(torch.concat([h_hard, x], axis=-1)) # Use relu because photon count always >= 0
        
        out = torch.concat([out_soft, out_medium, out_hard], axis=-1)
        # out = out # add a constant amount of count for smoother spectra (based on graphs in 3c50 paper)
        
        return out
    


"""################################################ 
    BERTground: Transformer-based model 
                for Tabular Data
################################################"""
"""------------------------------------------------
    Linear Tokenizer:
        Tokenize each numerical feature into N-D array
        'b x d' -> 'b x d x N'
------------------------------------------------"""
class FeatureEncoder(nn.Module):
    def __init__(self, n_features, embed_dim):
        super(FeatureEncoder, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(n_features, embed_dim))

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        return x * self.weights + self.biases
    
"""------------------------------------------------
    Group Linear Tokenizer
        Tokenize each group of features to two different N-D corresponding tokens
        E.g: Group 1 has d features: 'b x d' -> '(b x 5) x 2' when N=5
------------------------------------------------"""
class FeatureGroupTokenizer(nn.Module):
    def __init__(self, group_col_id, token_size):
        super(FeatureGroupTokenizer, self).__init__()
        
        self.group_col_id = group_col_id

        """ Separate weights for each Variable Group"""
        self.GroupEncoders = nn.ModuleDict({})
        for group in group_col_id:
            # Number of variables in each feature group is the input to the LinearEncoder
            num_vars = len(group_col_id[group])
            # Use sigmoid for one-hot encoding of tokens
            self.GroupEncoders[f'{group}'] = nn.Sequential(nn.Linear(num_vars, token_size), 
                                                           nn.Sigmoid())
            # self.GroupEncoders[f'{group} 2'] = nn.Linear(num_vars, token_size)
            
    def forward(self, x):
        """ List of group tokens"""
        token_list = []
        
        """ 
            Tokenize each feature group into 2 tokens
                -> 'b x g x token_size'
                where:
                    b = batch_size
                    g = number of feature groups
                    token_size
        """
        for group in self.group_col_id:
            col_ids = self.group_col_id[group]
            token = self.GroupEncoders[f'{group}'](x[:, col_ids])
            # token_2 = self.GroupEncoders[f'{group} 2'](x[:, col_ids])
            token_list.append(token)
            # token_list.append(token_2)
        
        """ Stack list of tokens and Reorder the axes"""
        tokens = torch.stack(token_list)
        # print(tokens.shape)
        tokens = torch.permute(tokens, (1,0,2))
        # print(tokens.shape)
            
        return tokens
            

# class NumericTransformer(nn.Module):
#     def __init__(self, input_dim, 
#                  output_dim, 
#                  group_col_id,
#                  token_size=4,
#                  nhead=4, 
#                  n_transformers=4, 
#                  embed_dim=16):
#         super(NumericTransformer, self).__init__()
#         """ Encode each feature group into two different N-D tokens"""
#         self.tokenizer = FeatureGroupTokenizer(group_col_id, token_size=token_size)
#         """ Transformer Encoder"""
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nhead,
#                                                         dropout=0.1, activation='gelu',
#                                                         dim_feedforward=embed_dim)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
#                                                          num_layers=n_transformers)
        
#         """ Prediction backbone"""
#         num_groups = len(group_col_id)
#         d = num_groups*token_size
#         # self.linear = nn.Linear(d, d)
#         # self.predictor = DNN(input_dim+d, output_dim)
#         # self.predictor = nn.Sequential(nn.Linear(input_dim + d, output_dim), nn.ReLU())
#         self.predictor = ResNet4(input_dim + d, output_dim)
        
#     def forward(self, x):
#         """ 1. Numerical feature encoder"""
#         tokens = self.tokenizer(x)
#         """ 2. Transformer cross-feature attention representation"""
#         h = self.transformer_encoder(tokens)
        
#         """ 3. Backbone ResNet"""
#         h = h.view((x.shape[0],-1))
#         # h = self.linear(h)
#         h = torch.concat([x, h], axis=-1)
#         out = self.predictor(h)
        
#         return out
    

"""------------------------------------------------
    BERTground:
        Consists of 3 components 
        1) Group Feature Tokenizer 
        2) Transformer Encoder with [CLS] token (BERT-like)
        3) ResNet as predictor
------------------------------------------------"""
class NumericContextualTransformer(nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 group_col_id,
                 token_size=4,
                 nhead=4, 
                 n_transformers=4, 
                 embed_dim=16):
        super(NumericContextualTransformer, self).__init__()
        """ Encode each feature group into two different N-D tokens"""
        self.tokenizer = FeatureGroupTokenizer(group_col_id, token_size=token_size)
        """ CLS token for contextualize numerical features"""
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_size))
        """ Transformer Encoder"""
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nhead,
                                                        dropout=0.1, activation='gelu',
                                                        dim_feedforward=embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                         num_layers=n_transformers)
        
        """ Prediction backbone"""
        num_groups = len(group_col_id)
        # d = num_groups*token_size
        self.predictor = ResNet4(input_dim + token_size, output_dim)
        
    def forward(self, x):
        bs = x.shape[0] # batch size
        
        """ 1. Numerical feature encoder"""
        tokens = self.tokenizer(x)
        """ 2. Transformer cross-feature attention representation 
            -> concat cls contextual token
        """
        cls_tokens = self.cls_token.repeat(bs, 1, 1)
        tokens = torch.cat((cls_tokens, tokens), dim = 1)
        h = self.transformer_encoder(tokens)
        """ 3. Retrieve cls_token"""
        h = h[:,0,:]
        """ 4. Backbone ResNet"""
        h = torch.concat([x, h], axis=-1)
        out = self.predictor(h)
        
        return out
    

"""################################################ 
    Frequency-reduced ResNet (Residual Network)
        + reduce the frequency of tabular dataset
            -> reduce the spectral bias of NN
################################################"""
class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)
    
class CustomLayer(nn.Module): # proposed layer
    def __init__(self, size):
        super(CustomLayer, self).__init__()
        
        self.size = size
        weights = torch.Tensor(torch.ones((1, size)) * 0.5)
        scaling_factors_src = torch.Tensor(torch.ones((1, size)))
        shifting_factors_src = torch.Tensor(torch.zeros((1, size)))
        scaling_factors_trg = torch.Tensor(torch.ones((1, size)))
        shifting_factors_trg = torch.Tensor(torch.zeros((1, size)))
        self.weights = nn.Parameter(weights)
        self.scaling_weights_src = nn.Parameter(scaling_factors_src)
        self.shifting_weights_src = nn.Parameter(shifting_factors_src)
        self.scaling_weights_trg = nn.Parameter(scaling_factors_trg)
        self.shifting_weights_trg = nn.Parameter(shifting_factors_trg)
        
    def forward(self, x):
        x_src = x.detach().clone() # raw input
        x_trg = x.detach().clone() # ranked input
        w = self.weights
        c_src = self.scaling_weights_src
        u_src = self.shifting_weights_src
        c_trg = self.scaling_weights_trg
        u_trg = self.shifting_weights_trg
        x = (((x_src + u_src) * c_src) * w) + (((x_trg + u_trg) * c_trg) * (1 - w))
        return x
    
class FR_DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FR_DNN, self).__init__()
        # frequency-reduced module: 
        # this is the core novelty of to reduce the frequency of tabular dataset
        self.customLayer = CustomLayer(input_dim)
        
        # this is the base model
        self.predictor = DNN4(input_dim, output_dim)
        
    def forward(self, x):
        x = self.customLayer(x)
        out = self.predictor(x)
        
        return out

    
class FR_ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FR_ResNet, self).__init__()
        # frequency-reduced module: 
        # this is the core novelty of to reduce the frequency of tabular dataset
        self.customLayer = CustomLayer(input_dim)
        
        # this is the base model
        self.hidden1 = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.resblock1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                       nn.Linear(hidden_dim, hidden_dim))
        self.resblock2 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), 
                                       nn.Linear(hidden_dim, hidden_dim))
        self.resblock3 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), 
                                       nn.Linear(hidden_dim, hidden_dim))
        self.resblock4 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), 
                                       nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Sequential(nn.Linear(hidden_dim+input_dim, output_dim), nn.ReLU())
        
    def forward(self, x):
        x = self.customLayer(x)
        
        h0 = self.hidden1(x)
        h1 = self.resblock1(h0)
        h2 = self.resblock2(torch.concat([h1+h0, x], axis=-1))
        h3 = self.resblock3(torch.concat([h2+h1, x], axis=-1))
        h4 = self.resblock4(torch.concat([h3+h2, x], axis=-1))
        out = self.output(torch.concat([h4+h3, x], axis=-1)) # Use relu because photon count always >= 0
        
        return out
    
"""------------------------------------------------
    Frequency-reduced BERTground:
        Consists of 3 components 
        0) Custom Module to reduce the dataset frequency
        1) Group Feature Tokenizer 
        2) Transformer Encoder with [CLS] token (BERT-like)
        3) ResNet as the predictor
------------------------------------------------"""
class FR_BERTground(nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 group_col_id,
                 token_size=4,
                 nhead=4, 
                 n_transformers=4, 
                 embed_dim=16):
        super(FR_BERTground, self).__init__()
        
        """ Encode each feature group into two different N-D tokens"""
        self.tokenizer = FeatureGroupTokenizer(group_col_id, token_size=token_size)
        """ CLS token for contextualize numerical features"""
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_size))
        """ Transformer Encoder"""
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nhead,
                                                        dropout=0.1, activation='gelu',
                                                        dim_feedforward=embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                         num_layers=n_transformers)
        
        """ frequency-reduced module: 
            this is the core novelty of to reduce the frequency of tabular dataset
        """
        self.customLayer = CustomLayer(input_dim)
        """ Prediction backbone"""
        num_groups = len(group_col_id)
        # d = num_groups*token_size
        self.predictor = ResNet4(input_dim + token_size, output_dim)
        
    def forward(self, x):
        bs = x.shape[0] # batch size
        
        """ 1. Numerical feature encoder"""
        tokens = self.tokenizer(x)
        """ 2. Transformer cross-feature attention representation 
            -> concat cls contextual token
        """
        cls_tokens = self.cls_token.repeat(bs, 1, 1)
        tokens = torch.cat((cls_tokens, tokens), dim = 1)
        h = self.transformer_encoder(tokens)
        """ 3. Retrieve cls_token"""
        h = h[:,0,:]
        """ 4. Backbone ResNet"""
        x = self.customLayer(x) # reduce dataset frequency of x
        h = torch.concat([x, h], axis=-1)
        out = self.predictor(h)
        
        return out

    
"""------------------------------------------------
        Neural Additive Model for NICER background
        Consists of 3 components 
        1) Group Feature Tokenizer 
        2) Transformer Encoder with [CLS] token (BERT-like)
        3) ResNet as predictor
------------------------------------------------"""
class NicerNAM(nn.Module):
    def __init__(self, input_dim, output_dim, group_col_id):
        super(NicerNAM, self).__init__()
        
        self.group_col_id = group_col_id
        
        """ Prediction backbone"""
        num_groups = len(group_col_id)
        self.AdditiveModels = nn.ModuleDict({})
        for group in group_col_id:
            # Number of variables in each feature group is the input to the LinearEncoder
            num_vars = len(group_col_id[group])
            # Use sigmoid for one-hot encoding of tokens
            self.AdditiveModels[f'{group}'] = ResNet(num_vars, output_dim, hidden_dim = 8 * num_vars)
        
    def forward(self, x):
        bs = x.shape[0] # batch size
        
        """ 
            1. Individual Neural Additive Models on each variable group
        """
        out = []
        for group in self.group_col_id:
            col_ids = self.group_col_id[group]
            out_ = self.AdditiveModels[f'{group}'](x[:, col_ids])
            out.append(out_)
            
        """ 
            2. Add predictions from indiviual additive neural networks
                Physically-motivated: 
                    the background can be divided into different background components
                    that are triggered by different variables (SCORPEON)
        """
        out = sum(out)
        
        return out
    
"""------------------------------------------------
    BERTground:
        Consists of 3 components 
        1) Group Feature Tokenizer 
        2) Transformer Encoder with [CLS] token (BERT-like)
        3) ResNet as predictor
------------------------------------------------"""
    
# class FTokenizer_ResNet4(nn.Module):
#     def __init__(self, input_dim, output_dim, group_col_id, token_size):
#         super(FTokenizer_ResNet4, self).__init__()
#         self.tokenizer =  FeatureGroupTokenizer(group_col_id, token_size=token_size)
        
#         num_groups = len(group_col_id)
#         self.backbone = ResNet4(input_dim + num_groups*token_size, output_dim)
        
#     def forward(self, x):
#         """ 1. Numerical feature encoder"""
#         tokens = self.tokenizer(x)
#         tokens = tokens.reshape((x.shape[0], -1))
#         out = self.backbone(torch.concat([x, tokens], axis=-1))
        
#         return out
    
""" 
    Initialization
        Xavier Uniform Initialization
"""
def init_DNN(m):
    torch.manual_seed(42)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)