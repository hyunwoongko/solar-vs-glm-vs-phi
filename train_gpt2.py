import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 256
n_layer = 12
n_head = 8
n_embd = 512
batch = 32
seq_len = 128
steps = 300
lr = 1e-3


def make_rule_tokens(step: int, stream_id: int):
    # Make simple arithmetic sequence
    start = (step + stream_id * 100) % vocab_size
    pos = torch.arange(seq_len, device=device, dtype=torch.long)
    seq = (start + pos) % vocab_size
    return seq.unsqueeze(0).expand(batch, seq_len).clone()


def reinit_layernorm(model: torch.nn.Module, mode: str, seed: int = 0):
    # Reinitialize LayerNorm weights according to the specified mode
    g = torch.Generator(device=device).manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            with torch.no_grad():
                if mode == "ones":
                    m.weight.fill_(1.0)
                elif mode == "rand":
                    m.weight.normal_(0.0, 1.0, generator=g)
                else:
                    raise ValueError(mode)
                if m.bias is not None:
                    m.bias.zero_()


def train_step_next_token(model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, stream_id: int):
    # Single training step for next-token prediction
    tokens = make_rule_tokens(step, stream_id)
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]
    opt.zero_grad()
    model.train()
    out = model(input_ids=input_ids)
    logits = out.logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    opt.step()
    return float(loss.item())


@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # Compute cosine similarity between two tensors
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return float(F.cosine_similarity(a, b, dim=0).item())


@torch.no_grad()
def gpt2_ln_cosines(m1: GPT2LMHeadModel, m2: GPT2LMHeadModel):
    # Compute cosine similarities of LayerNorm weights between two GPT-2 models
    n = len(m1.transformer.h)
    ln1 = [round(cosine(m1.transformer.h[i].ln_1.weight, m2.transformer.h[i].ln_1.weight), 5) for i in range(n)]
    ln2 = [round(cosine(m1.transformer.h[i].ln_2.weight, m2.transformer.h[i].ln_2.weight), 5) for i in range(n)]
    # use the follow lines if you wanna compare them too ;)
    # qkv_proj = [round(cosine(m1.transformer.h[i].attn.c_attn.weight, m2.transformer.h[i].attn.c_attn.weight), 5) for i in range(n)]
    # mlp = [round(cosine(m1.transformer.h[i].mlp.c_fc.weight, m2.transformer.h[i].mlp.c_fc.weight), 5) for i in range(n)]
    return {"ln_1": ln1, "ln_2": ln2}


def main():
    # Main training loop comparing different LayerNorm initializations
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=seq_len,
        n_ctx=seq_len,
        bos_token_id=0,
        eos_token_id=1,
    )

    torch.manual_seed(10)
    model_ones_a = GPT2LMHeadModel(config).to(device)
    torch.manual_seed(20)
    model_ones_b = GPT2LMHeadModel(config).to(device)
    torch.manual_seed(30)
    model_rand_a = GPT2LMHeadModel(config).to(device)
    torch.manual_seed(40)
    model_rand_b = GPT2LMHeadModel(config).to(device)

    reinit_layernorm(model_ones_a, "ones", seed=10)
    reinit_layernorm(model_ones_b, "ones", seed=20)
    reinit_layernorm(model_rand_a, "rand", seed=30)
    reinit_layernorm(model_rand_b, "rand", seed=40)

    opt_ones_a = torch.optim.Adam(model_ones_a.parameters(), lr=lr)
    opt_ones_b = torch.optim.Adam(model_ones_b.parameters(), lr=lr)
    opt_rand_a = torch.optim.Adam(model_rand_a.parameters(), lr=lr)
    opt_rand_b = torch.optim.Adam(model_rand_b.parameters(), lr=lr)

    losses = []
    for step in (pbar := tqdm(range(steps))):
        loss_1 = train_step_next_token(model_ones_a, opt_ones_a, step, stream_id=1)
        loss_2 = train_step_next_token(model_ones_b, opt_ones_b, step, stream_id=2)
        loss_3 = train_step_next_token(model_rand_a, opt_rand_a, step, stream_id=3)
        loss_4 = train_step_next_token(model_rand_b, opt_rand_b, step, stream_id=4)
        losses.append((loss_1, loss_2, loss_3, loss_4))
        pbar.set_postfix(loss1=f"{loss_1:.4f}", loss2=f"{loss_2:.4f}", loss3=f"{loss_3:.4f}", loss4=f"{loss_4:.4f}")

    cos_ones = gpt2_ln_cosines(model_ones_a, model_ones_b)
    cos_rand = gpt2_ln_cosines(model_rand_a, model_rand_b)

    print("-" * 50)
    print("First step losses:", [round(x, 3) for x in losses[0]])
    print("\nLast step losses:", [round(x, 3) for x in losses[-1]])
    print("\nCosines (ones init):", cos_ones)
    print("\nCosines (rand init):", cos_rand)
    print("-" * 50)


if __name__ == "__main__":
    main()
