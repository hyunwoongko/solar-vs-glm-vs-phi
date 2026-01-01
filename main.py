import os
import matplotlib
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device("cuda")

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
    }


def glm_params_by_layer_idx(idx):
    return {
        "input_layernorm": glm.layers[idx].input_layernorm.weight,
        "post_attention_layernorm": glm.layers[idx].post_attention_layernorm.weight,
    }


def phi_params_by_layer_idx(idx):
    return {
        "input_layernorm": phi.model.layers[idx].input_layernorm.weight,
        "post_attention_layernorm": phi.model.layers[idx].post_attention_layernorm.weight,
    }


@torch.no_grad()
def flat_param(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(device=device, dtype=dtype).contiguous().view(-1)


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


@torch.no_grad()
def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_diag(metric: str) -> float:
    if metric == "cosine":
        return 1.0
    if metric == "mean_abs_diff":
        return 0.0
    raise ValueError(f"unknown metric: {metric}")


@torch.no_grad()
def pair_metric(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    if metric == "cosine":
        return cosine_sim(a, b)
    if metric == "mean_abs_diff":
        return mean_abs_diff(a, b)
    raise ValueError(f"unknown metric: {metric}")


def infer_vmin_vmax(mats, params, metric: str, vmin, vmax):
    if metric == "cosine":
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
        out_path = os.path.join(out_dir, f"{filename_prefix}_{metric}_layer{layer_idx}.png")
        title = f"{metric} | model matrix | layer={layer_idx} (solar/glm/phi)"
        save_model_matrix_png(
            mats=mats, out_path=out_path, title=title, params=params, metric=metric, vmin=vmin, vmax=vmax, dpi=dpi
        )
        saved[layer_idx] = out_path
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
    out_path = os.path.join(
        out_dir, f"{filename_prefix}_{metric}_s{solar_layer_idx}_g{glm_layer_idx}_p{phi_layer_idx}.png"
    )
    title = f"{metric} | model matrix | solar={solar_layer_idx}, glm={glm_layer_idx}, phi={phi_layer_idx}"
    save_model_matrix_png(
        mats=mats, out_path=out_path, title=title, params=params, metric=metric, vmin=vmin, vmax=vmax, dpi=dpi
    )
    return out_path


@torch.no_grad()
def layer_to_layer_matrices_for_model(
    model_name: str,
    layer_indices,
    device: torch.device,
    dtype: torch.dtype,
    metric: str,
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
        model_name=model_name, layer_indices=layer_indices, device=device, dtype=dtype, metric=metric
    )
    layers_tag = "_".join(str(x) for x in layer_indices)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{metric}_{model_name}_layers{layers_tag}.png")
    title = f"{metric} | within-model layer matrix | model={model_name} | layers={layer_indices}"
    save_within_model_layer_matrix_png(
        mats=mats,
        layer_indices=layer_indices,
        out_path=out_path,
        title=title,
        params=params,
        metric=metric,
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
    )
    return out_path


def main():
    layers = [10, 20, 30]
    base_dir = "./outputs"
    conf_cos_dir = os.path.join(base_dir, "confusions", "cosine")
    conf_mad_dir = os.path.join(base_dir, "confusions", "mean_abs_diff")
    within_cos_dir = os.path.join(base_dir, "within_model", "cosine")
    within_mad_dir = os.path.join(base_dir, "within_model", "mean_abs_diff")
    params = ["input_layernorm", "post_attention_layernorm"]
    save_confusions_by_layer_list(
        layers=layers,
        out_dir=conf_cos_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="cosine",
    )
    save_confusion_for_mixed_layers(
        solar_layer_idx=10,
        glm_layer_idx=20,
        phi_layer_idx=30,
        out_dir=conf_cos_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="cosine",
    )
    save_within_model_layer_comparisons(
        model_name="solar",
        layer_indices=layers,
        out_dir=within_cos_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="cosine",
    )
    save_within_model_layer_comparisons(
        model_name="glm",
        layer_indices=layers,
        out_dir=within_cos_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="cosine",
    )
    save_within_model_layer_comparisons(
        model_name="phi",
        layer_indices=layers,
        out_dir=within_cos_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="cosine",
    )
    save_confusions_by_layer_list(
        layers=layers,
        out_dir=conf_mad_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="mean_abs_diff",
    )
    save_confusion_for_mixed_layers(
        solar_layer_idx=10,
        glm_layer_idx=20,
        phi_layer_idx=30,
        out_dir=conf_mad_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="mean_abs_diff",
    )
    save_within_model_layer_comparisons(
        model_name="solar",
        layer_indices=layers,
        out_dir=within_mad_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="mean_abs_diff",
    )
    save_within_model_layer_comparisons(
        model_name="glm",
        layer_indices=layers,
        out_dir=within_mad_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="mean_abs_diff",
    )
    save_within_model_layer_comparisons(
        model_name="phi",
        layer_indices=layers,
        out_dir=within_mad_dir,
        device=device,
        dtype=dtype,
        params=params,
        metric="mean_abs_diff",
    )


if __name__ == "__main__":
    main()
