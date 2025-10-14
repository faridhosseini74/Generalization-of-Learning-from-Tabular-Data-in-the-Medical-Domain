import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Dict, Any, Union, Tuple
from einops import rearrange

# Import traditional models
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    TRADITIONAL_AVAILABLE = True
except ImportError:
    print("Warning: Traditional models not available. Install sklearn, xgboost, lightgbm, catboost.")
    TRADITIONAL_AVAILABLE = False

# Transformer model configurations and classes
class ModelConfig:
    """Configuration class for model parameters."""
    
    def __init__(self, **kwargs):
        # Default parameters
        self.dim = kwargs.get('dim', 128)
        self.depth = kwargs.get('depth', 6)
        self.heads = kwargs.get('heads', 8)
        self.dim_head = kwargs.get('dim_head', 16)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.ff_dropout = kwargs.get('ff_dropout', 0.1)
        self.cont_embeddings = kwargs.get('cont_embeddings', 'MLP')
        
        # Model-specific parameters
        self.patch_len = kwargs.get('patch_len', 3)
        self.stride = kwargs.get('stride', 1)
        self.factor = kwargs.get('factor', 5)  # For Informer
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.
    Simplified implementation for glucose prediction.
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, dim) for cat_size in categories
        ]) if categories else nn.ModuleList()
        
        # Continuous feature processing
        self.cont_norm = nn.BatchNorm1d(num_continuous) if num_continuous > 0 else None
        self.cont_linear = nn.Linear(num_continuous, dim) if num_continuous > 0 else None
        
        # CLS token for final prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x_cat, x_cont, cat_mask=None, cont_mask=None):
        """Forward pass for sequence prediction."""
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        # Process each time step
        sequence_outputs = []
        
        for t in range(seq_len):
            tokens = []
            
            # Process categorical features
            if len(self.categories) > 0:
                x_cat_t = x_cat[:, t]  # [batch_size, num_categories]
                for i, embed_layer in enumerate(self.cat_embeddings):
                    cat_token = embed_layer(x_cat_t[:, i])  # [batch_size, dim]
                    tokens.append(cat_token.unsqueeze(1))  # [batch_size, 1, dim]
            
            # Process continuous features
            if self.num_continuous > 0:
                x_cont_t = x_cont[:, t]  # [batch_size, num_continuous]
                if self.cont_norm is not None:
                    x_cont_t = self.cont_norm(x_cont_t)
                cont_token = self.cont_linear(x_cont_t)  # [batch_size, dim]
                tokens.append(cont_token.unsqueeze(1))  # [batch_size, 1, dim]
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
            tokens.append(cls_tokens)
            
            # Concatenate all tokens
            if tokens:
                token_seq = torch.cat(tokens, dim=1)  # [batch_size, num_tokens, dim]
                
                # Apply transformer
                transformed = self.transformer(token_seq)  # [batch_size, num_tokens, dim]
                
                # Use CLS token output (last token)
                cls_output = transformed[:, -1]  # [batch_size, dim]
                sequence_outputs.append(cls_output)
        
        # Stack sequence outputs
        if sequence_outputs:
            sequence_tensor = torch.stack(sequence_outputs, dim=1)  # [batch_size, seq_len, dim]
            # Use last time step for prediction
            final_output = sequence_tensor[:, -1]  # [batch_size, dim]
        else:
            final_output = torch.zeros(batch_size, self.dim, device=device)
        
        # Generate prediction
        prediction = self.output_layer(final_output)
        return prediction


def get_available_transformer_models() -> List[str]:
    """Get list of available transformer model names."""
    return [
        'saint',
        'ft-transformer',
        'fttransformer',
        'tabpfn',
        'patchtst',
        'tst',
        'informer',
        'tabnet',
        'tabnetregressor'
    ]


def get_transformer_model_recommendations(
    sequence_length: int,
    num_features: int,
    dataset_size: int
) -> Dict[str, str]:
    """Get transformer model recommendations based on data characteristics."""
    recommendations = {}
    
    # Small datasets
    if dataset_size < 10000:
        recommendations['tabpfn'] = "TabPFN works well with small datasets due to prior-based learning"
        recommendations['tabnet'] = "TabNet is efficient for small-medium tabular data"
    
    # Medium to large datasets
    elif dataset_size < 100000:
        recommendations['saint'] = "SAINT combines tabular and sequential modeling effectively"
        recommendations['ft-transformer'] = "FT-Transformer is excellent for mixed tabular data"
        recommendations['tst'] = "TST provides good time series modeling capabilities"
    
    # Large datasets
    else:
        recommendations['informer'] = "Informer scales well to large sequences with efficient attention"
        recommendations['patchtst'] = "PatchTST is efficient for long time series"
        recommendations['saint'] = "SAINT handles large-scale tabular time series well"
    
    # Long sequences
    if sequence_length > 50:
        recommendations['informer'] = "Informer handles long sequences efficiently with sparse attention"
        recommendations['patchtst'] = "PatchTST uses patching for efficient long sequence processing"
    
    # Short sequences
    elif sequence_length < 10:
        recommendations['tabnet'] = "TabNet works well with limited temporal information"
        recommendations['ft-transformer'] = "FT-Transformer effective for short sequences"
    
    # High dimensional data
    if num_features > 50:
        recommendations['tabnet'] = "TabNet has built-in feature selection capabilities"
        recommendations['saint'] = "SAINT handles high-dimensional tabular data effectively"
    
    return recommendations


def create_transformer_model(
    model_name: str,
    categories: List[int],
    num_continuous: int,
    config: Optional[ModelConfig] = None,
    **kwargs
) -> nn.Module:
    """Create a transformer model."""
    model_name = model_name.lower()
    
    # Use provided config or create from kwargs
    cfg = config if config is not None else ModelConfig(**kwargs)
    
    if model_name in ['ft-transformer', 'fttransformer']:
        return FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout
        )
    elif model_name == 'saint':
        return SAINTForGlucose(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout,
            cont_embeddings=cfg.cont_embeddings
        )
    elif model_name == 'tabpfn':
        return TabPFN(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout
        )
    elif model_name == 'patchtst':
        return PatchTST(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout
        )
    elif model_name == 'tst':
        return TST(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout
        )
    elif model_name == 'informer':
        return Informer(
            categories=categories,
            num_continuous=num_continuous,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            factor=cfg.factor,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout
        )
    elif model_name in ['tabnet', 'tabnetregressor']:
        return TabNetRegressor(
            categories=categories,
            num_continuous=num_continuous,
            n_d=cfg.n_d,
            n_a=cfg.n_a,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            n_independent=cfg.n_independent,
            n_shared=cfg.n_shared,
            epsilon=cfg.epsilon,
            virtual_batch_size=cfg.virtual_batch_size,
            momentum=cfg.momentum
        )
    else:
        raise ValueError(f"Unknown transformer model: {model_name}")


# ============================================================================
# Additional Transformer Model Implementations
# ============================================================================

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Standard transformer block with attention and feed-forward."""
    
    def __init__(self, dim, heads, attn_dropout=0.1, ff_dropout=0.1, activation='gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ff_dropout)
        )
        
    def forward(self, x):
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class FeatureEmbedding(nn.Module):
    """Feature embedding for mixed categorical and continuous data."""
    
    def __init__(self, categories, num_continuous, dim):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Categorical embeddings
        if categories:
            self.num_unique_categories = sum(categories)
            self.cat_embed = nn.Embedding(self.num_unique_categories, dim)
            self.register_buffer('categories_offset', torch.tensor([0] + categories).cumsum(0)[:-1])
        
        # Continuous embeddings
        if num_continuous > 0:
            self.cont_embed = nn.Linear(num_continuous, dim)
        
        # Combination layer
        total_features = len(categories) + (1 if num_continuous > 0 else 0)
        if total_features > 1:
            self.combine = nn.Linear(dim * total_features, dim)
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        batch_size = x_cont.shape[0]
        device = x_cont.device
        
        # Categorical embedding
        if hasattr(self, 'cat_embed') and x_cat is not None and x_cat.shape[-1] > 0:
            x_cat_offset = x_cat + self.categories_offset
            cat_emb = self.cat_embed(x_cat_offset)  # [batch_size, n_cat, dim]
            embeddings.extend([cat_emb[:, i] for i in range(cat_emb.shape[1])])
        
        # Continuous embedding
        if hasattr(self, 'cont_embed') and x_cont.shape[-1] > 0:
            cont_emb = self.cont_embed(x_cont)  # [batch_size, dim]
            embeddings.append(cont_emb)
        
        # Handle case with no features
        if not embeddings:
            return torch.zeros(batch_size, self.dim, device=device)
        
        # Combine embeddings
        if len(embeddings) > 1 and hasattr(self, 'combine'):
            combined = torch.cat(embeddings, dim=-1)
            return self.combine(combined)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            # If multiple embeddings but no combine layer, average them
            return torch.stack(embeddings, dim=0).mean(dim=0)


# ============================================================================
# SAINT Model Components
# ============================================================================

class TabAttention(nn.Module):
    """
    Tabular Attention module for SAINT model
    """
    def __init__(
        self, 
        categories, 
        num_continuous, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        dim_out,
        mlp_hidden_mults = (4, 2),
        attn_dropout = 0., 
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        attentiontype = 'col',
        final_mlp_style = 'common'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        
        # Categories related variables
        self.categories = categories
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        
        # Continuous features related variables
        self.num_continuous = num_continuous
        
        # Other parameters
        self.dim = dim
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.cont_embeddings = cont_embeddings
        
        # Calculate category offsets
        self.register_buffer('categories_offset', torch.tensor([0] + categories).cumsum(0)[:-1])
        
        # Categorical embedding
        self.embeds = nn.Embedding(self.num_unique_categories, dim)
        
        # Mask embedding for handling missing categorical values
        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, dim)
        
        # Continuous embeddings
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                ) for _ in range(self.num_continuous)
            ])
            
        # Attention layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # First transformer block: Column/feature attention
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout),
                    nn.Dropout(ff_dropout)
                ),
                # Second transformer block: Feed-forward network
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * mlp_hidden_mults[0]),
                    nn.ReLU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(dim * mlp_hidden_mults[0], dim),
                    nn.Dropout(ff_dropout)
                )
            ]))
        
        # Output layer
        self.mlp_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_hidden_mults[0]),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mlp_hidden_mults[0], dim_out)
        )
        
    def forward(self, x_categ, x_cont, cat_enc=None, cont_enc=None):
        device = x_cont.device if x_cont.shape[-1] > 0 else x_categ.device
        batch_size = x_cont.shape[0] if x_cont.shape[-1] > 0 else x_categ.shape[0]
        
        # Process categorical features
        if self.num_categories > 0:
            if cat_enc is None:
                x_categ = x_categ + self.categories_offset
                x_categ_enc = self.embeds(x_categ)
                
                # Handle missing values using mask embeddings
                c_mask = torch.ones_like(x_categ, dtype=torch.bool)
                mask_embed_cat = self.mask_embeds_cat(
                    torch.arange(self.num_categories, device=device).repeat(batch_size, 1) * 2 + (~c_mask).long()
                )
                x_categ_enc = x_categ_enc + mask_embed_cat
            else:
                x_categ_enc = cat_enc
        else:
            x_categ_enc = torch.empty((batch_size, 0, self.dim), device=device)
        
        # Process continuous features
        if self.num_continuous > 0:
            if cont_enc is None:
                if self.cont_embeddings == 'MLP':
                    x_cont_enc = torch.zeros(batch_size, self.num_continuous, self.dim, device=device)
                    for i in range(self.num_continuous):
                        x_cont_enc[:, i] = self.simple_MLP[i](x_cont[:, i:i+1])
                    
                    # Handle missing values using mask embeddings
                    c_mask = torch.ones_like(x_cont, dtype=torch.bool)
                    mask_indices = torch.arange(self.num_continuous, device=device).repeat(batch_size, 1) * 2 + (~c_mask).long()
                    mask_embed_cont = self.mask_embeds_cont(mask_indices)
                    
                    x_cont_enc = x_cont_enc + mask_embed_cont
                else:
                    raise ValueError(f"Unknown continuous embedding type: {self.cont_embeddings}")
            else:
                x_cont_enc = cont_enc
        else:
            x_cont_enc = torch.empty((batch_size, 0, self.dim), device=device)
        
        # Combine categorical and continuous embeddings
        if x_categ_enc.shape[1] > 0 and x_cont_enc.shape[1] > 0:
            x = torch.cat((x_categ_enc, x_cont_enc), dim=1)
        elif x_categ_enc.shape[1] > 0:
            x = x_categ_enc
        else:
            x = x_cont_enc
        
        # Pass through attention layers
        for attn, ff in self.layers:
            # Attention block
            x_ln = attn[0](x)
            x_attn, _ = attn[1](x_ln.permute(1, 0, 2), x_ln.permute(1, 0, 2), x_ln.permute(1, 0, 2))
            x = x + attn[2](x_attn.permute(1, 0, 2))
            
            # Feed-forward block
            x = x + ff(x)
        
        # Output projection
        if self.attentiontype == 'col':
            y = x.mean(dim=1)
            return self.mlp_out(y)
        else:
            return self.mlp_out(x.mean(dim=1))


class SAINTForGlucose(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim=128,
        depth=6,
        heads=8,
        dim_head=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        cont_embeddings='MLP'
    ):
        super().__init__()
        
        # TabAttention for processing tabular data features
        self.tab_attention = TabAttention(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dim_out=dim,
            mlp_hidden_mults=(4, 2),
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            cont_embeddings=cont_embeddings,
            attentiontype='col'
        )
        
        # Additional sequence processing layer for time series data
        self.sequence_layer = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=2,
            batch_first=True,
            dropout=ff_dropout if depth > 1 else 0
        )
        
        # Final prediction layer
        self.final_layer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        if patient_mask is None:
            patient_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Process each time step with TabAttention
        tab_outputs = []
        
        for t in range(seq_len):
            x_cat_t = x_categ[:, t] if x_categ.shape[-1] > 0 else torch.empty((batch_size, 0), device=device)
            x_cont_t = x_cont[:, t]
            
            tab_out = self.tab_attention(x_cat_t, x_cont_t)
            tab_outputs.append(tab_out)
        
        # Stack outputs
        tab_sequence = torch.stack(tab_outputs, dim=1)
        tab_sequence = tab_sequence * patient_mask.unsqueeze(-1).float()
        
        # Process with sequence layer
        seq_out, _ = self.sequence_layer(tab_sequence)
        
        # Use last valid position
        if patient_mask.all():
            last_out = seq_out[:, -1]
        else:
            last_valid_indices = patient_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(batch_size, device=device)
            last_out = seq_out[batch_indices, last_valid_indices]
        
        return self.final_layer(last_out)


# ============================================================================
# TabPFN Model
# ============================================================================

class TabPFN(nn.Module):
    """
    Simplified TabPFN-style model for tabular data.
    Uses prior-fitting transformer architecture.
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Feature embedding
        self.feature_embed = FeatureEmbedding(categories, num_continuous, dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=ff_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Prior prediction head
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        # Process each timestep
        embeddings = []
        
        for t in range(seq_len):
            x_cat_t = x_categ[:, t] if x_categ.shape[-1] > 0 else None
            x_cont_t = x_cont[:, t]
            embed_t = self.feature_embed(x_cat_t, x_cont_t)
            embeddings.append(embed_t)
        
        x = torch.stack(embeddings, dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        
        # Use last valid timestep
        if patient_mask is not None:
            last_valid_indices = patient_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(batch_size, device=device)
            final_hidden = x[batch_indices, last_valid_indices]
        else:
            final_hidden = x[:, -1]
        
        return self.predictor(final_hidden)


# ============================================================================
# PatchTST Model
# ============================================================================

class PatchTST(nn.Module):
    """
    PatchTST model adapted for tabular time series.
    Patching approach for time series forecasting.
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        patch_len: int = 3,
        stride: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        self.patch_len = patch_len
        self.stride = stride
        
        # Feature embedding for each timestep
        self.feature_embed = FeatureEmbedding(categories, num_continuous, dim)
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_len * dim, dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        # Get feature embeddings for each timestep
        embeddings = []
        
        for t in range(seq_len):
            x_cat_t = x_categ[:, t] if x_categ.shape[-1] > 0 else None
            x_cont_t = x_cont[:, t]
            embed_t = self.feature_embed(x_cat_t, x_cont_t)
            embeddings.append(embed_t)
        
        x = torch.stack(embeddings, dim=1)
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len].flatten(start_dim=1)
            patch_embed = self.patch_embed(patch)
            patches.append(patch_embed)
        
        if not patches:
            patches = [x.mean(dim=1)]
        
        x = torch.stack(patches, dim=1)
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        x = self.norm(x)
        
        return self.head(x)


# ============================================================================
# TST Model
# ============================================================================

class TST(nn.Module):
    """
    Time Series Transformer with learnable positional encoding.
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Feature embedding
        self.feature_embed = FeatureEmbedding(categories, num_continuous, dim)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 100, dim))  # Max 100 timesteps
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        # Get feature embeddings
        embeddings = []
        
        for t in range(seq_len):
            x_cat_t = x_categ[:, t] if x_categ.shape[-1] > 0 else None
            x_cont_t = x_cont[:, t]
            embed_t = self.feature_embed(x_cat_t, x_cont_t)
            embeddings.append(embed_t)
        
        x = torch.stack(embeddings, dim=1)
        x = x + self.pos_embed[:, :seq_len]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Use last valid timestep
        if patient_mask is not None:
            last_valid_indices = patient_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(batch_size, device=device)
            final_hidden = x[batch_indices, last_valid_indices]
        else:
            final_hidden = x[:, -1]
        
        return self.head(self.norm(final_hidden))


# ============================================================================
# Informer Model
# ============================================================================

class Informer(nn.Module):
    """
    Informer model with ProbSparse self-attention.
    Simplified version for glucose prediction.
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        factor: int = 5,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Feature embedding
        self.feature_embed = FeatureEmbedding(categories, num_continuous, dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim)
        
        # Informer encoder layers
        self.layers = nn.ModuleList([
            InformerLayer(dim, heads, factor, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        device = x_cont.device
        
        # Get feature embeddings
        embeddings = []
        
        for t in range(seq_len):
            x_cat_t = x_categ[:, t] if x_categ.shape[-1] > 0 else None
            x_cont_t = x_cont[:, t]
            embed_t = self.feature_embed(x_cat_t, x_cont_t)
            embeddings.append(embed_t)
        
        x = torch.stack(embeddings, dim=1)
        x = self.pos_encoding(x)
        
        # Apply Informer layers
        for layer in self.layers:
            x = layer(x)
        
        # Use last valid timestep
        if patient_mask is not None:
            last_valid_indices = patient_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(batch_size, device=device)
            final_hidden = x[batch_indices, last_valid_indices]
        else:
            final_hidden = x[:, -1]
        
        return self.head(self.norm(final_hidden))


class InformerLayer(nn.Module):
    """Informer layer with ProbSparse attention."""
    
    def __init__(self, dim, heads, factor=5, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ProbSparseAttention(dim, heads, factor, attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ff_dropout)
        )
        
    def forward(self, x):
        # Self-attention
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class ProbSparseAttention(nn.Module):
    """Simplified ProbSparse attention mechanism."""
    
    def __init__(self, dim, heads, factor=5, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.factor = factor
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Simplified sparse attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply attention sparsity
        k_sparse = min(self.factor * int(math.log(max(seq_len, 2))), seq_len)
        top_dots, top_indices = dots.topk(k_sparse, dim=-1)
        
        # Create sparse attention weights
        attn_weights = torch.zeros_like(dots)
        attn_weights.scatter_(-1, top_indices, F.softmax(top_dots, dim=-1))
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


# ============================================================================
# TabNet Model
# ============================================================================

class TabNetRegressor(nn.Module):
    """
    TabNet implementation for regression tasks.
    Based on "TabNet: Attentive Interpretable Tabular Learning"
    """
    
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Calculate input dimension
        input_dim = len(categories) + num_continuous
        self.input_dim = input_dim
        
        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
        # Step-specific layers
        self.steps = nn.ModuleList([
            TabNetStep(input_dim, n_d, n_a, gamma, epsilon)
            for _ in range(n_steps)
        ])
        
        # Final classifier
        self.final_mapping = nn.Linear(n_d, 1)
        
    def forward(self, x_categ, x_cont, cat_mask=None, con_mask=None, patient_mask=None):
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]
        
        # Process last timestep
        if patient_mask is not None:
            last_valid_indices = patient_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(batch_size, device=x_cont.device)
            x_cat_last = x_categ[batch_indices, last_valid_indices] if x_categ.shape[-1] > 0 else None
            x_cont_last = x_cont[batch_indices, last_valid_indices]
        else:
            x_cat_last = x_categ[:, -1] if x_categ.shape[-1] > 0 else None
            x_cont_last = x_cont[:, -1]
        
        # Combine categorical and continuous features
        features = []
        if x_cat_last is not None and x_cat_last.shape[-1] > 0:
            features.append(x_cat_last.float())
        if x_cont_last.shape[-1] > 0:
            features.append(x_cont_last)
        
        if features:
            x = torch.cat(features, dim=-1)
        else:
            x = torch.zeros(batch_size, self.input_dim, device=x_cont.device)
        
        # Initial processing
        x = self.initial_bn(x)
        
        # TabNet steps
        prior = torch.ones(batch_size, self.input_dim, device=x.device)
        step_outputs = []
        
        for step in self.steps:
            x_out, prior = step(x, prior)
            step_outputs.append(x_out)
        
        # Combine step outputs
        output = torch.stack(step_outputs, dim=0).sum(dim=0)
        
        return self.final_mapping(output)


class TabNetStep(nn.Module):
    """Single step in TabNet architecture."""
    
    def __init__(self, input_dim, n_d, n_a, gamma, epsilon):
        super().__init__()
        self.n_d = n_d
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Attention transformer
        self.attention_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        
        # Feature transformer
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, n_d * 2),
            nn.BatchNorm1d(n_d * 2)
        )
        
    def forward(self, x, prior):
        # Attention
        mask = self.attention_transform(x)
        sparse_mask = F.softmax(mask * prior, dim=-1)
        
        # Feature transformation
        output = F.glu(self.feature_transform(x * sparse_mask), dim=-1)
        
        # Update prior
        prior = prior * (self.gamma - sparse_mask)
        
        return output, prior


# ============================================================================
# Traditional Model Wrapper
# ============================================================================

class TraditionalModelWrapper(nn.Module):
    """
    Wrapper for traditional sklearn models to make them compatible 
    with the PyTorch training pipeline.
    """
    def __init__(self, model_name, categories, num_continuous, **kwargs):
        super().__init__()
        self.model_name = model_name.lower()
        self.categories = categories
        self.num_continuous = num_continuous
        self.fitted = False
        self.scaler = StandardScaler() if TRADITIONAL_AVAILABLE else None
        
        if not TRADITIONAL_AVAILABLE:
            raise ImportError("Traditional models not available. Please install required packages.")
        
        # Create the actual sklearn model
        self.model = self._create_sklearn_model(**kwargs)
        
        # These will be set during training
        self.feature_names = None
        self.n_features = None
        
    def _create_sklearn_model(self, **kwargs):
        """Create the underlying sklearn model with optimized parameters."""
        random_state = kwargs.get('random_state', 42)
        n_jobs = kwargs.get('n_jobs', -1)
        
        if self.model_name in ['randomforest', 'rf']:
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 8),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 4),
                max_features=kwargs.get('max_features', 'sqrt'),
                random_state=random_state,
                n_jobs=n_jobs
            )
        elif self.model_name in ['xgboost', 'xgb']:
            return XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.08),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                reg_alpha=kwargs.get('reg_alpha', 0.5),
                reg_lambda=kwargs.get('reg_lambda', 2.0),
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0
            )
        elif self.model_name in ['lightgbm', 'lgb']:
            return LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.08),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                reg_alpha=kwargs.get('reg_alpha', 0.5),
                reg_lambda=kwargs.get('reg_lambda', 2.0),
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1
            )
        elif self.model_name in ['catboost', 'cat']:
            return CatBoostRegressor(
                iterations=kwargs.get('iterations', kwargs.get('n_estimators', 150)),
                depth=kwargs.get('depth', kwargs.get('max_depth', 5)),
                learning_rate=kwargs.get('learning_rate', 0.08),
                l2_leaf_reg=kwargs.get('l2_leaf_reg', 5.0),
                random_state=random_state,
                verbose=False,
                thread_count=kwargs.get('thread_count', n_jobs)
            )
        elif self.model_name in ['mlp', 'neural_network']:
            return MLPRegressor(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (32, 16)),
                alpha=kwargs.get('alpha', 0.5),
                learning_rate=kwargs.get('learning_rate', 'adaptive'),
                max_iter=kwargs.get('max_iter', 300),
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown traditional model: {self.model_name}")
    
    def _prepare_features(self, cat_features, cont_features, statistics=None):
        """Prepare flattened features for sklearn models."""
        if isinstance(cat_features, torch.Tensor):
            cat_features = cat_features.cpu().numpy()
        if isinstance(cont_features, torch.Tensor):
            cont_features = cont_features.cpu().numpy()
        if statistics is not None and isinstance(statistics, torch.Tensor):
            statistics = statistics.cpu().numpy()
            
        batch_size = cat_features.shape[0]
        features = []
        
        # Flatten categorical features
        if cat_features.size > 0:
            cat_flat = cat_features.reshape(batch_size, -1)
            features.append(cat_flat)
        
        # Flatten continuous features
        if cont_features.size > 0:
            cont_flat = cont_features.reshape(batch_size, -1)
            features.append(cont_flat)
            
        # Add statistics if provided
        if statistics is not None and statistics.size > 0:
            stats_flat = statistics.reshape(batch_size, -1)
            features.append(stats_flat)
        
        # Concatenate all features
        if features:
            X = np.concatenate(features, axis=1)
        else:
            X = np.zeros((batch_size, 1))
        
        # Handle NaN values and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Remove constant features to avoid numerical issues
        if X.shape[1] > 1:
            feature_variance = np.var(X, axis=0)
            non_constant_mask = feature_variance > 1e-8
            if np.any(non_constant_mask):
                X = X[:, non_constant_mask]
        
        return X.astype(np.float32)
    
    def fit(self, train_loader, val_loader=None):
        """Fit the traditional model using data from the train_loader."""
        # Collect all training data
        X_list, y_list = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:  # With statistics
                cat_features, cont_features, statistics, targets = batch
            else:  # Without statistics
                cat_features, cont_features, targets = batch
                statistics = None
            
            X_batch = self._prepare_features(cat_features, cont_features, statistics)
            y_batch = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        
        # Store feature info
        self.n_features = X_train.shape[1]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit the model
        self.model.fit(X_train_scaled, y_train)
        self.fitted = True
        
        # Calculate training loss for compatibility
        train_pred = self.model.predict(X_train_scaled)
        train_loss = np.mean((train_pred - y_train) ** 2)
        
        result = {'train_loss': float(train_loss)}
        
        # If validation data is provided, calculate validation loss
        if val_loader is not None:
            val_loss = self.evaluate(val_loader)
            result['val_loss'] = val_loss
            
        return result
    
    def evaluate(self, data_loader):
        """Evaluate the model on the given data loader."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")
            
        X_list, y_list = [], []
        
        for batch in data_loader:
            if len(batch) == 4:
                cat_features, cont_features, statistics, targets = batch
            else:
                cat_features, cont_features, targets = batch
                statistics = None
            
            X_batch = self._prepare_features(cat_features, cont_features, statistics)
            y_batch = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        X_val = np.vstack(X_list)
        y_val = np.concatenate(y_list)
        
        X_val_scaled = self.scaler.transform(X_val)
        predictions = self.model.predict(X_val_scaled)
        
        loss = np.mean((predictions - y_val) ** 2)
        return float(loss)
    
    def forward(self, cat_features, cont_features, statistics=None):
        """Forward pass - used for inference."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before inference")
        
        X = self._prepare_features(cat_features, cont_features, statistics)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return torch.tensor(predictions, dtype=torch.float32, device=cat_features.device)
    
    def predict(self, cat_features, cont_features, statistics=None):
        """Prediction method for compatibility."""
        return self.forward(cat_features, cont_features, statistics)
    
    def __call__(self, *args, **kwargs):
        """Make the model callable."""
        return self.forward(*args, **kwargs)


class ModelFactory:
    """
    Unified model factory for creating different architectures.
    Supports Transformer models and Traditional ML models.
    """
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of all available model architectures."""
        models = []
        
        # Add transformer models
        models.extend(get_available_transformer_models())
        
        # Add traditional models
        if TRADITIONAL_AVAILABLE:
            models.extend(['randomforest', 'xgboost', 'lightgbm', 'catboost', 'mlp'])
            
        return models
    
    @staticmethod
    def create_model(
        model_name: str,
        categories: List[int],
        num_continuous: int,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create a model based on the specified architecture.
        
        Args:
            model_name: Name of the model architecture
            categories: List of category dimensions for categorical features
            num_continuous: Number of continuous features
            config: Optional configuration dictionary
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model
        """
        model_name = model_name.lower()
        
        # Merge config with kwargs
        if config:
            merged_config = {**config, **kwargs}
        else:
            merged_config = kwargs
        
        if TRADITIONAL_AVAILABLE and model_name in ['randomforest', 'xgboost', 'lightgbm', 'catboost', 'mlp']:
            # Create traditional model wrapper
            return TraditionalModelWrapper(
                model_name=model_name,
                categories=categories,
                num_continuous=num_continuous,
                **merged_config
            )
        
        elif model_name in get_available_transformer_models():
            # Create transformer model using the integrated factory
            model_config = ModelConfig(**merged_config) if merged_config else None
            return create_transformer_model(
                model_name=model_name,
                categories=categories,
                num_continuous=num_continuous,
                config=model_config
            )
        
        else:
            available = ModelFactory.get_available_models()
            raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model architecture."""
        model_name = model_name.lower()
        
        if model_name in get_available_transformer_models():
            # Get transformer model info
            descriptions = {
                'saint': {
                    'full_name': 'Self-Attention and Intersample Attention Network',
                    'type': 'transformer',
                    'description': 'Combines row and column attention for tabular data',
                    'good_for': ['tabular_data', 'time_series', 'mixed_features']
                },
                'ft-transformer': {
                    'full_name': 'Feature Tokenizer Transformer',
                    'type': 'transformer', 
                    'description': 'Transformer that tokenizes features for tabular learning',
                    'good_for': ['tabular_data', 'categorical_features', 'feature_importance']
                },
                'fttransformer': {
                    'full_name': 'Feature Tokenizer Transformer',
                    'type': 'transformer', 
                    'description': 'Transformer that tokenizes features for tabular learning',
                    'good_for': ['tabular_data', 'categorical_features', 'feature_importance']
                },
                'tabpfn': {
                    'full_name': 'TabPFN',
                    'type': 'transformer',
                    'description': 'Prior-fitting transformer for small tabular datasets',
                    'good_for': ['small_datasets', 'tabular_data', 'few_shot_learning']
                },
                'patchtst': {
                    'full_name': 'PatchTST',
                    'type': 'transformer',
                    'description': 'Patching-based time series transformer',
                    'good_for': ['long_sequences', 'time_series', 'efficiency']
                },
                'tst': {
                    'full_name': 'Time Series Transformer',
                    'type': 'transformer',
                    'description': 'Transformer with learnable positional encodings for time series',
                    'good_for': ['time_series', 'temporal_patterns', 'sequential_data']
                },
                'informer': {
                    'full_name': 'Informer',
                    'type': 'transformer',
                    'description': 'Efficient transformer with ProbSparse attention',
                    'good_for': ['long_sequences', 'large_datasets', 'efficiency']
                },
                'tabnet': {
                    'full_name': 'TabNet',
                    'type': 'attention_based',
                    'description': 'Attentive interpretable tabular learning',
                    'good_for': ['tabular_data', 'feature_selection', 'interpretability']
                },
                'tabnetregressor': {
                    'full_name': 'TabNet Regressor',
                    'type': 'attention_based',
                    'description': 'Attentive interpretable tabular learning for regression',
                    'good_for': ['tabular_data', 'feature_selection', 'interpretability']
                }
            }
            
            info = descriptions.get(model_name, {})
            info['name'] = model_name.upper()
            return info
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_recommendations(
        sequence_length: int,
        num_features: int,
        dataset_size: int,
        has_missing_values: bool = False
    ) -> Dict[str, str]:
        """Get model recommendations based on data characteristics."""
        recommendations = {}
        
        # Add transformer recommendations
        transformer_recs = get_transformer_model_recommendations(sequence_length, num_features, dataset_size)
        recommendations.update(transformer_recs)
        
        # Additional recommendations based on missing values
        if has_missing_values:
            recommendations['saint'] = "SAINT can handle missing values in tabular data effectively"
            recommendations['tabnet'] = "TabNet has built-in mechanisms for handling missing data"
        
        return recommendations


def get_available_models_list() -> List[str]:
    """Get available models - convenience function."""
    return ModelFactory.get_available_models()
