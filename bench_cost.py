"""
bench_cost.py -- measured cost of the benchmark models (tables/cost.tex).

For each model in bench.py and each input type (a 64 px image, or a clip of
T=8 frames): parameter count, FLOPs per forward pass (torch's own
FlopCounterMode, which counts matrix multiplies and convolutions), and
single-thread CPU latency at batch size 1, median and 90th percentile of 50
timed passes after 10 warm-up passes.  Latencies depend on the machine;
the paper's were measured on the workstation described in the README.

    python bench_cost.py
"""

import json
import time

import numpy as np
import torch

import bench

torch.set_num_threads(1)


def count_flops(net, x, head):
    """FLOPs per forward pass (2 x multiply-accumulates).

    torch's FlopCounterMode counts matmuls and convolutions but not the fused
    kernels behind nn.LSTM, and in eval mode nn.TransformerEncoder takes a
    fused fast path it also cannot see.  So the pass is run in train mode
    (which disables the fast path; dropout does not change FLOPs) and every
    nn.LSTM adds its gate cost analytically: 2 * 4H(I+H) per step per layer.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as e:                       # report, never invent
        print("  flop count unavailable for", head, e)
        return None
    extra = [0]

    def lstm_hook(mod, inp, out):
        z = inp[0]
        B, T = (z.shape[0], z.shape[1]) if mod.batch_first else (z.shape[1], z.shape[0])
        H, I = mod.hidden_size, mod.input_size
        for l in range(mod.num_layers):
            i_l = I if l == 0 else H * (2 if mod.bidirectional else 1)
            extra[0] += 2 * 4 * H * (i_l + H) * T * B * (2 if mod.bidirectional else 1)

    hooks = [m.register_forward_hook(lstm_hook) for m in net.modules()
             if isinstance(m, torch.nn.LSTM)]
    was = net.training
    net.train()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    net.train(was)
    for h in hooks:
        h.remove()
    return int(fc.get_total_flops()) + extra[0]


def measure(head, nc, video):
    net = bench.Net("cnn" if head == "proposed" else head, nc, video).eval()
    x = torch.rand(1, 8, 3, 64, 64) if video else torch.rand(1, 3, 64, 64)
    params = int(sum(p.numel() for p in net.parameters()))
    flops = count_flops(net, x, head)
    with torch.no_grad():
        for _ in range(10):
            net(x)
        ts = []
        for _ in range(50):
            t0 = time.perf_counter(); net(x); ts.append(1e3 * (time.perf_counter() - t0))
    return {"params": params, "flops": flops,
            "latency_ms_median": float(np.median(ts)),
            "latency_ms_p90": float(np.percentile(ts, 90))}


def flops_only():
    """Recount FLOPs into an existing results/bench_cost.json, keeping its
    latencies (which were measured with nothing else running)."""
    out = json.load(open("results/bench_cost.json"))
    for kind, video, nc in (("image", False, 35), ("video", True, 97)):
        for m in out[kind]:
            net = bench.Net(m, nc, video)
            x = torch.rand(1, 8, 3, 64, 64) if video else torch.rand(1, 3, 64, 64)
            out[kind][m]["flops"] = count_flops(net, x, m)
            print("%-6s %-7s %d FLOPs" % (kind, m, out[kind][m]["flops"]))
    out["flops_note"] = "train-mode pass plus analytic nn.LSTM gate cost"
    json.dump(out, open("results/bench_cost.json", "w"), indent=1)
    print("updated results/bench_cost.json")


def main():
    out = {"threads": 1, "batch": 1, "image": {}, "video": {}}
    for kind, video, nc in (("image", False, 35), ("video", True, 97)):
        for m in bench.MODELS:
            if m == "proposed":                  # identical network to cnn at inference
                continue
            out[kind][m] = measure(m, nc, video)
            r = out[kind][m]
            print("%-6s %-7s %8d params  %s FLOPs  %.2f ms" % (
                kind, m, r["params"], r["flops"], r["latency_ms_median"]))
    json.dump(out, open("results/bench_cost.json", "w"), indent=1)
    print("wrote results/bench_cost.json")


if __name__ == "__main__":
    import sys
    flops_only() if "--flops-only" in sys.argv else main()
