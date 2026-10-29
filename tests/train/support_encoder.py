import torch

class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding.from_pretrained(torch.eye(12), freeze=False)

    def forward(self, input_ids):
        return torch.nn.functional.normalize(self.table(input_ids[:, 0]), dim=-1)

def tokenize(titles, **kwargs):
    return {'input_ids': torch.arange(len(titles))[:, None]}
