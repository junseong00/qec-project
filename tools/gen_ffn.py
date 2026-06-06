"""
INT8 quantization + golden integer model + Verilog codegen for the QECCT-Student
TransformerBlock FFN:   Linear(32->128) -> GELU -> Linear(128->32)   (per token)

Same bit-exact strategy as gen_noise_estimator.py:
  1. Quantize weights to INT8 (symmetric per-tensor). Inputs are real activations
     (norm2 outputs) -> quantized to INT8 with a calibrated symmetric scale.
  2. GOLDEN INTEGER model = exactly the integer ops the RTL performs
     (int32 MAC, round-half-up requant M*acc>>S, INT8 activations, 256-entry GELU LUT).
  3. RTL mirrors those ops -> RTL == golden bit-exact (checked in xsim).
  4. Separately report INT8-vs-float accuracy of the FFN block.

Unlike noise_estimator there is NO final sigmoid: the FFN output is a signed INT8
activation (scale_out) that feeds the residual add in the full block.

Outputs (BLOCK selects which transformer block's FFN, default 0):
  rtl/ffn_params.vh        - weights/biases/GELU-LUT/requant consts
  rtl/tb_ffn_vectors.vh    - input INT8 vectors + golden INT8 outputs
  golden/golden_ffn_report.txt - accuracy summary
"""
import os, sys, math
import numpy as np
import torch

BLOCK   = int(os.environ.get("FFN_BLOCK", "0"))
S_SHIFT = 16
NVEC    = 100                      # bit-exact test vectors emitted for the TB
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tools" else HERE
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
GOLDEN_DIR = os.path.join(ROOT, "golden")
RTL  = os.path.join(ROOT, "rtl")
os.makedirs(RTL, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)
sys.path.insert(0, MODEL_DIR)

from qecct_models import SurfaceCode
from qecct_student import QECCTStudent

# ---------------------------------------------------------------- model + weights
code  = SurfaceCode(3)
model = QECCTStudent(code, N=2, d_model=32, n_heads=4, use_faulty=True)
sd    = torch.load(os.path.join(MODEL_DIR, "student_L3_faulty.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(sd); model.eval()

W1 = sd[f"blocks.{BLOCK}.ffn.0.weight"].numpy().astype(np.float64)   # (128,32)
b1 = sd[f"blocks.{BLOCK}.ffn.0.bias"].numpy().astype(np.float64)     # (128,)
W2 = sd[f"blocks.{BLOCK}.ffn.2.weight"].numpy().astype(np.float64)   # (32,128)
b2 = sd[f"blocks.{BLOCK}.ffn.2.bias"].numpy().astype(np.float64)     # (32,)
DFF, DM = W1.shape          # 128, 32
DOUT, _ = W2.shape          # 32, 128
print(f"FFN block {BLOCK}: FC1 {W1.shape}  FC2 {W2.shape}")

# ---------------------------------------------------------------- capture FFN inputs (norm2 out)
caps = []
def hook(mod, inp, out):
    caps.append(out.detach().reshape(-1, out.shape[-1]).numpy())
model.blocks[BLOCK].norm2.register_forward_hook(hook)

allx  = np.array([[int(c) for c in f"{v:08b}"] for v in range(256)], dtype=np.float32)
syn20 = np.load(os.path.join(DATA_DIR, "fpga_test_vectors.npz"))["syndrome"].astype(np.float32)
with torch.no_grad():
    model(torch.tensor(allx))     # 256 calibration syndromes
    n_cal = len(np.concatenate(caps, 0))
    model(torch.tensor(syn20))    # 20 official test vectors
Xin = np.concatenate(caps, 0).astype(np.float64)      # (4692, 32) all tokens
Xtest = Xin[n_cal:]                                   # (340, 32) from the 20 test vectors
print(f"captured {Xin.shape} FFN-input tokens ({n_cal} cal + {len(Xtest)} test)")

# ---------------------------------------------------------------- float FFN reference
def float_ffn(x):                 # x: (B,32) float
    xt = torch.tensor(x, dtype=torch.float32)
    y1 = xt @ torch.tensor(W1.T, dtype=torch.float32) + torch.tensor(b1, dtype=torch.float32)
    h  = torch.nn.functional.gelu(y1)
    y2 = h @ torch.tensor(W2.T, dtype=torch.float32) + torch.tensor(b2, dtype=torch.float32)
    return y1.numpy(), h.numpy(), y2.numpy()

cy1, ch, cy2 = float_ffn(Xin)     # calibrate over the full pool

# ---------------------------------------------------------------- quant scales (symmetric)
scale_x  = np.abs(Xin).max() / 127.0
scale_w1 = np.abs(W1).max()  / 127.0
scale_w2 = np.abs(W2).max()  / 127.0
scale_y1 = np.abs(cy1).max() / 127.0     # pre-GELU
scale_h  = np.abs(ch).max()  / 127.0     # post-GELU (FC2 input)
scale_out= np.abs(cy2).max() / 127.0     # FFN output activation

def qsym(arr, scale, lo=-127, hi=127):
    return np.clip(np.round(arr / scale), lo, hi).astype(np.int64)

W1q = qsym(W1, scale_w1)                                    # (128,32)
W2q = qsym(W2, scale_w2)                                    # (32,128)
b1q = np.round(b1 / (scale_x * scale_w1)).astype(np.int64)  # acc1 scale
b2q = np.round(b2 / (scale_h * scale_w2)).astype(np.int64)  # acc2 scale

def make_M(ratio):
    M = int(round(ratio * (1 << S_SHIFT)))
    assert 0 < M < (1 << 31), f"M out of range: {M}"
    return M
M1 = make_M(scale_x * scale_w1 / scale_y1)
M2 = make_M(scale_h * scale_w2 / scale_out)

def requant(acc, M):
    t = int(acc) * M + (1 << (S_SHIFT - 1))
    q = t >> S_SHIFT
    return max(-128, min(127, q))

# ---------------------------------------------------------------- GELU LUT (idx = q+128)
gelu_lut = np.zeros(256, dtype=np.int64)
for i in range(256):
    q = i - 128
    h_real = float(torch.nn.functional.gelu(torch.tensor(q * scale_y1)).item())
    gelu_lut[i] = int(max(-128, min(127, round(h_real / scale_h))))

# ---------------------------------------------------------------- golden INTEGER model
def golden_int(xq):               # xq: (32,) int8
    xq = [int(v) for v in xq]
    y1q = np.empty(DFF, dtype=np.int64)
    for j in range(DFF):
        acc = int(b1q[j]) + sum(int(W1q[j, i]) * xq[i] for i in range(DM))
        y1q[j] = requant(acc, M1)
    hq = np.array([gelu_lut[int(y1q[j]) + 128] for j in range(DFF)], dtype=np.int64)
    y2q = np.empty(DOUT, dtype=np.int64)
    for k in range(DOUT):
        acc = int(b2q[k]) + sum(int(W2q[k, j]) * int(hq[j]) for j in range(DFF))
        y2q[k] = requant(acc, M2)
    return y2q

# ---------------------------------------------------------------- accuracy (full pool)
Xq_all = qsym(Xin, scale_x)
gold_all = np.array([golden_int(Xq_all[i]) for i in range(len(Xq_all))], dtype=np.int64)
gold_deq = gold_all * scale_out
max_err  = np.abs(gold_deq - cy2).max()
rms_err  = np.sqrt(np.mean((gold_deq - cy2) ** 2))
denom    = np.abs(cy2).mean()
rel_rms  = rms_err / denom
# sign-agreement (relevant since FFN output feeds a residual add)
sign_agree = (np.sign(gold_deq) == np.sign(cy2)).mean()

# ---------------------------------------------------------------- pick NVEC TB vectors (good coverage)
# include per-dimension max/min-abs tokens (datapath extremes) + even spread
idx = set()
for col in range(DM):
    idx.add(int(np.argmax(Xin[:, col]))); idx.add(int(np.argmin(Xin[:, col])))
idx.add(int(np.argmax(np.abs(Xin).sum(1)))); idx.add(int(np.argmin(np.abs(Xin).sum(1))))
spread = np.linspace(0, len(Xin) - 1, NVEC, dtype=int).tolist()
sel = list(dict.fromkeys(list(idx) + spread))[:NVEC]
sel = sorted(sel)
Xq_sel = Xq_all[sel]
gold_sel = gold_all[sel]

report = []
def log(s): print(s); report.append(s)
log("=" * 68)
log(f"INT8 FFN (block {BLOCK}) quantization report   Linear(32->128)->GELU->Linear(128->32)")
log("=" * 68)
log(f"scales: x={scale_x:.6g} w1={scale_w1:.6g} y1={scale_y1:.6g} h={scale_h:.6g} "
    f"w2={scale_w2:.6g} out={scale_out:.6g}")
log(f"requant: S={S_SHIFT}  M1={M1}  M2={M2}")
log(f"W1q range [{W1q.min()},{W1q.max()}]  W2q range [{W2q.min()},{W2q.max()}]")
log(f"b1q range [{b1q.min()},{b1q.max()}]  b2q range [{b2q.min()},{b2q.max()}]")
log(f"x  in [{Xq_all.min()},{Xq_all.max()}]   y2q out in [{gold_all.min()},{gold_all.max()}]")
log("")
log(f"[{len(Xin)} tokens]  INT8 vs float FFN:")
log(f"   max|INT8-float|   = {max_err:.4f}   (output abs-mean = {denom:.4f})")
log(f"   RMS error         = {rms_err:.4f}   (relative RMS = {rel_rms*100:.2f}%)")
log(f"   sign agreement    = {sign_agree*100:.2f}%")
log(f"[TB] emitting {len(sel)} bit-exact vectors (extremes + even spread)")
with open(os.path.join(GOLDEN_DIR, "golden_ffn_report.txt"), "w") as f:
    f.write("\n".join(report) + "\n")

# ================================================================ Verilog codegen
def hx8(v):  return f"{v & 0xFF:02x}"
def sd32(v): return (f"-32'sd{-int(v)}" if v < 0 else f"32'sd{int(v)}")

L = []; A = L.append
A("// AUTO-GENERATED by gen_ffn.py -- do not edit by hand")
A(f"// INT8 FFN params (transformer block {BLOCK}). requant: (M*acc + (1<<{S_SHIFT-1})) >>> {S_SHIFT}")
A(f"localparam integer DM   = {DM};")
A(f"localparam integer DFF  = {DFF};")
A(f"localparam integer DOUT = {DOUT};")
A(f"localparam integer SSH  = {S_SHIFT};")
A(f"localparam signed [31:0] M1 = 32'sd{M1};")
A(f"localparam signed [31:0] M2 = 32'sd{M2};")
A("")
CHK = DFF // DM                 # FC2 chunks per neuron (128/32 = 4)
A("// Flat weight arrays (used by area-min ffn.v baseline only):")
A("reg signed [7:0]  W1 [0:DFF*DM-1];")
A("reg signed [31:0] B1 [0:DFF-1];")
A("reg signed [7:0]  W2 [0:DOUT*DFF-1];")
A("reg signed [31:0] B2 [0:DOUT-1];")
A("reg signed [7:0]  GELU_LUT [0:255];")
A("// Packed wide-word weight ROM (used by pipelined ffn_pipe.v):")
A("//   one 256-bit word = 32 int8 weights (lane s = bits [8s +: 8])")
A("//   addr 0..DFF-1          : FC1, indexed by neuron")
A("//   addr DFF..DFF+DOUT*CHK : FC2, indexed by (DFF + neuron*CHK + chunk)")
A(f"localparam integer CHK   = {CHK};")
A(f"localparam integer WROM_N = {DFF + DOUT*CHK};")
A("reg [255:0] WROM [0:WROM_N-1];")
A("")
A("initial begin")
for j in range(DFF):
    for i in range(DM):
        A(f"  W1[{j*DM+i}] = 8'sh{hx8(int(W1q[j,i]))};")
for j in range(DFF):
    A(f"  B1[{j}] = {sd32(b1q[j])};")
for k in range(DOUT):
    for j in range(DFF):
        A(f"  W2[{k*DFF+j}] = 8'sh{hx8(int(W2q[k,j]))};")
for k in range(DOUT):
    A(f"  B2[{k}] = {sd32(b2q[k])};")
for i in range(256):
    A(f"  GELU_LUT[{i}] = 8'sh{hx8(int(gelu_lut[i]))};")
# packed WROM: FC1 by neuron (0..DFF-1)
for j in range(DFF):
    word = 0
    for i in range(DM):
        word |= (int(W1q[j, i]) & 0xFF) << (8 * i)
    A(f"  WROM[{j}] = 256'h{word:064x};")
# packed WROM: FC2 by (DFF + neuron*CHK + chunk)
for k in range(DOUT):
    for c in range(CHK):
        word = 0
        for s in range(DM):
            word |= (int(W2q[k, c * DM + s]) & 0xFF) << (8 * s)
        A(f"  WROM[{DFF + k*CHK + c}] = 256'h{word:064x};")
A("end")
with open(os.path.join(RTL, "ffn_params.vh"), "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote rtl/ffn_params.vh ({len(L)} lines)")

# ---- TB vectors: NVEC input vectors (DM int8 each) + golden output (DOUT int8 each)
T = []; B = T.append
B(f"// AUTO-GENERATED FFN test vectors (block {BLOCK}): input INT8 + golden INT8 output")
B(f"localparam integer NVEC = {len(sel)};")
B("reg signed [7:0] TB_IN  [0:NVEC*DM-1];      // input activation vectors (int8)")
B("reg signed [7:0] TB_OUT [0:NVEC*DOUT-1];    // expected golden FFN output (int8)")
B("initial begin")
for v in range(len(sel)):
    for i in range(DM):
        B(f"  TB_IN[{v*DM+i}] = 8'sh{hx8(int(Xq_sel[v,i]))};")
for v in range(len(sel)):
    for k in range(DOUT):
        B(f"  TB_OUT[{v*DOUT+k}] = 8'sh{hx8(int(gold_sel[v,k]))};")
B("end")
with open(os.path.join(RTL, "tb_ffn_vectors.vh"), "w") as f:
    f.write("\n".join(T) + "\n")
print("wrote rtl/tb_ffn_vectors.vh")
print("DONE")
