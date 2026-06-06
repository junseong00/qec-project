"""Probe FFN input (norm2 output) and output activations from the real model,
to determine INT8 scales before building the golden/RTL FFN block."""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tools" else HERE
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, MODEL_DIR)

from qecct_models import SurfaceCode
from qecct_student import QECCTStudent

code = SurfaceCode(3)
model = QECCTStudent(code, N=2, d_model=32, n_heads=4, use_faulty=True)
sd = torch.load(os.path.join(MODEL_DIR, "student_L3_faulty.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(sd)
model.eval()

# Capture FFN input (= norm2 output) and FFN output per block via hooks
caps = {}
def mk(name):
    def hook(mod, inp, out):
        caps.setdefault(name, []).append(out.detach().reshape(-1, out.shape[-1]).numpy())
    return hook
for b in range(2):
    model.blocks[b].norm2.register_forward_hook(mk(f"in{b}"))   # FFN input
    model.blocks[b].ffn.register_forward_hook(mk(f"out{b}"))    # FFN output

# Drive: all 256 syndromes (calibration) + 20 test vectors
allx = np.array([[int(c) for c in f"{v:08b}"] for v in range(256)], dtype=np.float32)
d = np.load(os.path.join(DATA_DIR, "fpga_test_vectors.npz"))
syn20 = d["syndrome"].astype(np.float32)
with torch.no_grad():
    model(torch.tensor(allx))
    model(torch.tensor(syn20))

for b in range(2):
    xin = np.concatenate(caps[f"in{b}"], 0)
    xout = np.concatenate(caps[f"out{b}"], 0)
    print(f"block {b}: FFN in  shape {xin.shape}  range [{xin.min():.4f},{xin.max():.4f}]  maxabs {np.abs(xin).max():.4f}")
    print(f"block {b}: FFN out shape {xout.shape}  range [{xout.min():.4f},{xout.max():.4f}]  maxabs {np.abs(xout).max():.4f}")

# Weight ranges
for b in range(2):
    W1 = sd[f"blocks.{b}.ffn.0.weight"].numpy(); b1 = sd[f"blocks.{b}.ffn.0.bias"].numpy()
    W2 = sd[f"blocks.{b}.ffn.2.weight"].numpy(); b2 = sd[f"blocks.{b}.ffn.2.bias"].numpy()
    print(f"block {b}: W1 maxabs {np.abs(W1).max():.4f}  b1 maxabs {np.abs(b1).max():.4f}  "
          f"W2 maxabs {np.abs(W2).max():.4f}  b2 maxabs {np.abs(b2).max():.4f}")
