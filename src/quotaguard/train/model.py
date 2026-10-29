import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from .data import DEFAULT_MAX_LENGTH, TITLE_ONLY, TextOptionsTokenizer, validate_text_options

def validate_model_options(config):
    if config.get('kind', 'lora') not in {'lora', 'residual_e5'}:
        raise ValueError('model.kind must be lora or residual_e5')
    if config.get('pooling', 'cls') not in {'cls', 'mean'}:
        raise ValueError('model.pooling must be cls or mean')
    if config.get('text_prefix', '') not in {'', 'query: '}:
        raise ValueError('model.text_prefix must be empty or "query: "')
    try:
        validate_text_options(config.get('text_fields', TITLE_ONLY), config.get('max_length', DEFAULT_MAX_LENGTH))
    except ValueError as error:
        raise ValueError(f'model.{error}') from None
    if config.get('kind') == 'residual_e5':
        if config.get('pooling', 'mean') != 'mean' or config.get('text_prefix', 'query: ') != 'query: ':
            raise ValueError('residual_e5 requires mean pooling and symmetric "query: " prefix')
        rank = config.get('residual_rank', 8)
        if type(rank) is not int or rank < 1:
            raise ValueError('model.residual_rank must be a positive integer')

class PrefixedTokenizer:

    def __init__(self, tokenizer, prefix):
        self.tokenizer, self.prefix = (tokenizer, prefix)

    def __call__(self, titles, *args, **kwargs):
        texts = self.prefix + titles if isinstance(titles, str) else [self.prefix + title for title in titles]
        return self.tokenizer(texts, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

def build_tokenizer(config):
    from transformers import BertTokenizerFast
    validate_model_options(config)
    tokenizer = BertTokenizerFast.from_pretrained(config['name'], cache_dir=config['cache_dir'])
    prefix = config.get('text_prefix', 'query: ' if config.get('kind') == 'residual_e5' else '')
    tokenizer = PrefixedTokenizer(tokenizer, prefix) if prefix else tokenizer
    options = validate_text_options(config.get('text_fields', TITLE_ONLY), config.get('max_length', DEFAULT_MAX_LENGTH))
    if options != (TITLE_ONLY, DEFAULT_MAX_LENGTH):
        tokenizer = TextOptionsTokenizer(tokenizer, *options)
    return tokenizer

class TitleEncoder(nn.Module):

    def __init__(self, bert, pooling='cls'):
        super().__init__()
        validate_model_options({'pooling': pooling})
        self.bert = bert
        self.pooling = pooling

    def forward(self, **tokens):
        hidden = self.bert(**tokens).last_hidden_state
        if self.pooling == 'cls':
            return F.normalize(hidden[:, 0], dim=-1)
        mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

class ResidualE5(nn.Module):

    def __init__(self, encoder, dimension, *, rank=8, seed):
        super().__init__()
        from .samplers import generator_for
        self.encoder = encoder.requires_grad_(False).eval()
        self.A = nn.Parameter(torch.randn(dimension, rank, generator=generator_for(seed, 'residual_init')) * 0.02)
        self.B = nn.Parameter(torch.zeros(dimension, rank))
        self._public_cache = {}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def query_embeddings(self, embeddings):
        with torch.autocast(embeddings.device.type, enabled=False):
            return embeddings + embeddings @ self.A @ self.B.T

    def forward(self, input_ids, *, cached_embeddings):
        return torch.cat((self.query_embeddings(cached_embeddings[:1]), cached_embeddings[1:]), dim=0)

    @torch.no_grad()
    def encode_public(self, tokens, batch_size):
        digest = hashlib.sha256()
        for name, value in sorted(tokens.items()):
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        key = (digest.hexdigest(), str(self.A.device))
        if key not in self._public_cache:
            self._public_cache[key] = encode_entities(self.encoder, tokens, batch_size).detach()
        return self._public_cache[key]

def checkpoint_payload(model, config):
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    if isinstance(model, ResidualE5):
        weights['config'] = {k: v for k, v in config.items() if k != 'device'}
    return weights

def residual_checkpoint_weights(payload, config, *, dimension=768):
    if not isinstance(payload, dict) or set(payload) != {'A', 'B', 'config'}:
        raise ValueError('Invalid residual checkpoint keys')
    if payload['config'] != {k: v for k, v in config.items() if k != 'device'}:
        raise ValueError('Residual checkpoint configuration mismatch')
    shape = (dimension, config['model'].get('residual_rank', 8))
    for key in ('A', 'B'):
        weight = payload[key]
        if not isinstance(weight, torch.Tensor) or weight.shape != shape or weight.dtype != torch.float32 or (not torch.isfinite(weight).all()):
            raise ValueError('Invalid residual checkpoint factor')
    return {key: payload[key] for key in ('A', 'B')}

def build_model(config, *, seed):
    from peft import LoraConfig, get_peft_model
    from transformers import BertModel
    validate_model_options(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        bert = BertModel.from_pretrained(config['name'], cache_dir=config['cache_dir'], add_pooling_layer=False, attn_implementation='eager')
        for module in bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        if config.get('kind') == 'residual_e5':
            model = ResidualE5(TitleEncoder(bert, pooling='mean'), bert.config.hidden_size, rank=config.get('residual_rank', 8), seed=seed)
        else:
            lora = LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'], target_modules=config['target_modules'], lora_dropout=0.0, bias='none')
            model = TitleEncoder(get_peft_model(bert, lora), pooling=config.get('pooling', 'cls'))
    tokenizer = build_tokenizer(config)
    return (model, tokenizer)

@torch.no_grad()
def encode_entities(model, tokens, batch_size, bf16=False):
    if batch_size < 1:
        raise ValueError('Encoding batch size must be positive')
    if isinstance(model, ResidualE5):
        return model.encode_public(tokens, batch_size)
    training = model.training
    model.eval()
    try:
        device_type = next(model.parameters()).device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
            return torch.cat([model(**{k: v[i:i + batch_size] for k, v in tokens.items()}).float() for i in range(0, len(tokens['input_ids']), batch_size)])
    finally:
        model.train(training)
