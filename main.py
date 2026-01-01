import os
import json
import matplotlib
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device("cuda")
eps = 1e-12

solar = AutoModel.from_pretrained(
    "upstage/Solar-Open-100B",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
glm = AutoModel.from_pretrained(
    "zai-org/GLM-4.5-Air",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
phi = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-MoE-instruct",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)


def solar_params_by_layer_idx(idx):
    return {
        "input_layernorm": solar.layers[idx].input_layernorm.weight,
        "post_attention_layernorm": solar.layers[idx].post_attention_layernorm.weight,
        "k_proj": solar.layers[idx].self_attn.k_proj.weight,
        "v_proj": solar.layers[idx].self_attn.v_proj.weight,
    }


def glm_params_by_layer_idx(idx):
    return {
        "input_layernorm": glm.layers[idx].input_layernorm.weight,
        "post_attention_layernorm": glm.layers[idx].post_attention_layernorm.weight,
        "k_proj": glm.layers[idx].self_attn.k_proj.weight,
        "v_proj": glm.layers[idx].self_attn.v_proj.weight,
    }


def phi_params_by_layer_idx(idx):
    return {
        "input_layernorm": phi.model.layers[idx].input_layernorm.weight,
        "post_attention_layernorm": phi.model.layers[idx].post_attention_layernorm.weight,
        "k_proj": phi.model.layers[idx].self_attn.k_proj.weight,
        "v_proj": phi.model.layers[idx].self_attn.v_proj.weight,
    }


@torch.no_grad()
def flat_param(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(device=device, dtype=dtype).contiguous().view(-1)


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


@torch.no_grad()
def centered_cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a - a.mean()
    b = b - b.mean()
    na = a.norm().clamp_min(eps)
    nb = b.norm().clamp_min(eps)
    return (a @ b / (na * nb)).item()


@torch.no_grad()
def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a - a.mean()
    b = b - b.mean()
    sa = a.std(unbiased=False).clamp_min(eps)
    sb = b.std(unbiased=False).clamp_min(eps)
    return ((a / sa) * (b / sb)).mean().item()


@torch.no_grad()
def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


@torch.no_grad()
def rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    denom = a.norm().clamp_min(eps)
    return ((a - b).norm() / denom).item()


@torch.no_grad()
def cv_diff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    cva = a.std(unbiased=False) / a.mean().abs().clamp_min(eps)
    cvb = b.std(unbiased=False) / b.mean().abs().clamp_min(eps)
    return torch.abs(cva - cvb).item()


@torch.no_grad()
def p99_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = torch.quantile(torch.abs(a), torch.tensor(0.99, device=a.device, dtype=a.dtype))
    pb = torch.quantile(torch.abs(b), torch.tensor(0.99, device=b.device, dtype=b.dtype))
    return torch.abs(pa - pb).item()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_similarity_metric(metric: str) -> bool:
    return metric in ["cosine", "centered_cosine", "pearson"]


def default_diag(metric: str) -> float:
    if is_similarity_metric(metric):
        return 1.0
    return 0.0


@torch.no_grad()
def pair_metric(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    if metric == "cosine":
        return cosine_sim(a, b)
    if metric == "centered_cosine":
        return centered_cosine_sim(a, b, eps=eps)
    if metric == "pearson":
        return pearson_corr(a, b, eps=eps)
    if metric == "mean_abs_diff":
        return mean_abs_diff(a, b)
    if metric == "rel_l2":
        return rel_l2(a, b, eps=eps)
    if metric == "cv_diff":
        return cv_diff(a, b, eps=eps)
    if metric == "p99_abs_diff":
        return p99_abs_diff(a, b)
    raise ValueError(f"unknown metric: {metric}")


def infer_vmin_vmax(mats, params, metric: str, vmin, vmax):
    if is_similarity_metric(metric):
        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0
        return vmin, vmax
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        mx = 0.0
        for k in params:
            mx = max(mx, float(mats[k].max().item()))
        vmax = mx if mx > 0 else 1e-12
    return vmin, vmax


@torch.no_grad()
def confusion_matrices_for_layers(
    solar_layer_idx: int,
    glm_layer_idx: int,
    phi_layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    metric: str,
):
    solar_params = solar_params_by_layer_idx(solar_layer_idx)
    glm_params = glm_params_by_layer_idx(glm_layer_idx)
    phi_params = phi_params_by_layer_idx(phi_layer_idx)
    out = {}
    diag = default_diag(metric)
    for param_name in solar_params.keys():
        s = flat_param(solar_params[param_name], device, dtype)
        g = flat_param(glm_params[param_name], device, dtype)
        p = flat_param(phi_params[param_name], device, dtype)
        m = torch.empty(3, 3, device="cpu", dtype=torch.float32)
        m[0, 0] = diag
        m[1, 1] = diag
        m[2, 2] = diag
        m[0, 1] = pair_metric(s, g, metric)
        m[1, 0] = m[0, 1]
        m[0, 2] = pair_metric(s, p, metric)
        m[2, 0] = m[0, 2]
        m[1, 2] = pair_metric(g, p, metric)
        m[2, 1] = m[1, 2]
        out[param_name] = m
    return out


def mats_to_jsonable(mats):
    return {k: [[float(x) for x in row] for row in v.tolist()] for k, v in mats.items()}


def save_json(obj, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_model_matrix_png(
    mats,
    out_path: str,
    title: str,
    params=None,
    metric: str = "cosine",
    vmin=None,
    vmax=None,
    dpi: int = 200,
):
    model_labels = ["solar", "glm", "phi"]
    if params is None:
        params = list(mats.keys())
    vmin, vmax = infer_vmin_vmax(mats, params, metric, vmin, vmax)
    ncols = len(params)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.6), squeeze=False)
    axes = axes[0]
    for j, param_name in enumerate(params):
        ax = axes[j]
        mat = mats[param_name].numpy()
        im = ax.imshow(mat, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(model_labels, rotation=30, ha="right")
        ax.set_yticklabels(model_labels)
        ax.set_title(param_name)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{mat[r, c]:.6g}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_confusions_by_layer_list(
    layers,
    out_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    params=None,
    metric: str = "cosine",
    vmin=None,
    vmax=None,
    dpi: int = 200,
    filename_prefix: str = "confusion",
):
    ensure_dir(out_dir)
    if params is None:
        sample = solar_params_by_layer_idx(layers[0])
        params = list(sample.keys())
    saved = {}
    for layer_idx in layers:
        mats = confusion_matrices_for_layers(
            solar_layer_idx=layer_idx,
            glm_layer_idx=layer_idx,
            phi_layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            metric=metric,
        )
        out_png = os.path.join(out_dir, f"{filename_prefix}_{metric}_layer{layer_idx}.png")
        out_json = os.path.join(out_dir, f"{filename_prefix}_{metric}_layer{layer_idx}.json")
        title = f"{metric} | model matrix | layer={layer_idx} (solar/glm/phi)"
        save_model_matrix_png(
            mats=mats, out_path=out_png, title=title, params=params, metric=metric, vmin=vmin, vmax=vmax, dpi=dpi
        )
        save_json(
            {
                "metric": metric,
                "layer": int(layer_idx),
                "models": ["solar", "glm", "phi"],
                "mats": mats_to_jsonable(mats),
            },
            out_json,
        )
        saved[layer_idx] = {"png": out_png, "json": out_json}
    return saved


def save_confusion_for_mixed_layers(
    solar_layer_idx: int,
    glm_layer_idx: int,
    phi_layer_idx: int,
    out_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    params=None,
    metric: str = "cosine",
    vmin=None,
    vmax=None,
    dpi: int = 200,
    filename_prefix: str = "confusion_mixed",
):
    ensure_dir(out_dir)
    if params is None:
        sample = solar_params_by_layer_idx(solar_layer_idx)
        params = list(sample.keys())
    mats = confusion_matrices_for_layers(
        solar_layer_idx=solar_layer_idx,
        glm_layer_idx=glm_layer_idx,
        phi_layer_idx=phi_layer_idx,
        device=device,
        dtype=dtype,
        metric=metric,
    )
    out_png = os.path.join(
        out_dir, f"{filename_prefix}_{metric}_s{solar_layer_idx}_g{glm_layer_idx}_p{phi_layer_idx}.png"
    )
    out_json = os.path.join(
        out_dir, f"{filename_prefix}_{metric}_s{solar_layer_idx}_g{glm_layer_idx}_p{phi_layer_idx}.json"
    )
    title = f"{metric} | model matrix | solar={solar_layer_idx}, glm={glm_layer_idx}, phi={phi_layer_idx}"
    save_model_matrix_png(
        mats=mats, out_path=out_png, title=title, params=params, metric=metric, vmin=vmin, vmax=vmax, dpi=dpi
    )
    save_json(
        {
            "metric": metric,
            "solar_layer": int(solar_layer_idx),
            "glm_layer": int(glm_layer_idx),
            "phi_layer": int(phi_layer_idx),
            "models": ["solar", "glm", "phi"],
            "mats": mats_to_jsonable(mats),
        },
        out_json,
    )
    return {"png": out_png, "json": out_json}


@torch.no_grad()
def layer_to_layer_matrices_for_model(
    model_name: str, layer_indices, device: torch.device, dtype: torch.dtype, metric: str
):
    if model_name == "solar":
        getter = solar_params_by_layer_idx
    elif model_name == "glm":
        getter = glm_params_by_layer_idx
    elif model_name == "phi":
        getter = phi_params_by_layer_idx
    else:
        raise ValueError(f"unknown model_name: {model_name}")
    L = len(layer_indices)
    params0 = getter(layer_indices[0])
    param_names = list(params0.keys())
    flat = {k: [] for k in param_names}
    for idx in layer_indices:
        p = getter(idx)
        for k in param_names:
            flat[k].append(flat_param(p[k], device, dtype))
    out = {}
    diag = default_diag(metric)
    for k in param_names:
        m = torch.empty(L, L, device="cpu", dtype=torch.float32)
        for i in range(L):
            m[i, i] = diag
        for i in range(L):
            for j in range(i + 1, L):
                v = pair_metric(flat[k][i], flat[k][j], metric)
                m[i, j] = v
                m[j, i] = v
        out[k] = m
    return out


def save_within_model_layer_matrix_png(
    mats,
    layer_indices,
    out_path: str,
    title: str,
    params=None,
    metric: str = "cosine",
    vmin=None,
    vmax=None,
    dpi: int = 200,
):
    if params is None:
        params = list(mats.keys())
    labels = [str(x) for x in layer_indices]
    L = len(layer_indices)
    vmin, vmax = infer_vmin_vmax(mats, params, metric, vmin, vmax)
    ncols = len(params)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
    axes = axes[0]
    for j, param_name in enumerate(params):
        ax = axes[j]
        mat = mats[param_name].numpy()
        im = ax.imshow(mat, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(param_name)
        for r in range(L):
            for c in range(L):
                ax.text(c, r, f"{mat[r, c]:.6g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_within_model_layer_comparisons(
    model_name: str,
    layer_indices,
    out_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    params=None,
    metric: str = "cosine",
    vmin=None,
    vmax=None,
    dpi: int = 200,
    filename_prefix: str = "within",
):
    ensure_dir(out_dir)
    if params is None:
        if model_name == "solar":
            sample = solar_params_by_layer_idx(layer_indices[0])
        elif model_name == "glm":
            sample = glm_params_by_layer_idx(layer_indices[0])
        elif model_name == "phi":
            sample = phi_params_by_layer_idx(layer_indices[0])
        else:
            raise ValueError(f"unknown model_name: {model_name}")
        params = list(sample.keys())
    mats = layer_to_layer_matrices_for_model(
        model_name=model_name,
        layer_indices=layer_indices,
        device=device,
        dtype=dtype,
        metric=metric,
    )
    layers_tag = "_".join(str(x) for x in layer_indices)
    out_png = os.path.join(out_dir, f"{filename_prefix}_{metric}_{model_name}_layers{layers_tag}.png")
    out_json = os.path.join(out_dir, f"{filename_prefix}_{metric}_{model_name}_layers{layers_tag}.json")
    title = f"{metric} | within-model layer matrix | model={model_name} | layers={layer_indices}"
    save_within_model_layer_matrix_png(
        mats=mats,
        layer_indices=layer_indices,
        out_path=out_png,
        title=title,
        params=params,
        metric=metric,
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
    )
    save_json(
        {
            "metric": metric,
            "model": model_name,
            "layers": [int(x) for x in layer_indices],
            "params": list(mats.keys()),
            "mats": mats_to_jsonable(mats),
        },
        out_json,
    )
    return {"png": out_png, "json": out_json}


def main():
    layers = [10, 20, 30]
    base_dir = "./outputs"
    params = ["input_layernorm", "post_attention_layernorm", "k_proj", "v_proj"]
    metrics = ["cosine", "centered_cosine", "pearson", "mean_abs_diff", "rel_l2", "cv_diff", "p99_abs_diff"]
    for metric in metrics:
        conf_dir = os.path.join(base_dir, "confusions", metric)
        within_dir = os.path.join(base_dir, "within_model", metric)
        save_confusions_by_layer_list(
            layers=layers,
            out_dir=conf_dir,
            device=device,
            dtype=dtype,
            params=params,
            metric=metric,
        )
        save_confusion_for_mixed_layers(
            solar_layer_idx=10,
            glm_layer_idx=20,
            phi_layer_idx=30,
            out_dir=conf_dir,
            device=device,
            dtype=dtype,
            params=params,
            metric=metric,
        )
        save_within_model_layer_comparisons(
            model_name="solar",
            layer_indices=layers,
            out_dir=within_dir,
            device=device,
            dtype=dtype,
            params=params,
            metric=metric,
        )
        save_within_model_layer_comparisons(
            model_name="glm",
            layer_indices=layers,
            out_dir=within_dir,
            device=device,
            dtype=dtype,
            params=params,
            metric=metric,
        )
        save_within_model_layer_comparisons(
            model_name="phi",
            layer_indices=layers,
            out_dir=within_dir,
            device=device,
            dtype=dtype,
            params=params,
            metric=metric,
        )


if __name__ == "__main__":
    main()
