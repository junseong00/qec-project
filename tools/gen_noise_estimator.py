"""
INT8 quantization + golden integer model + Verilog codegen for QECCT-Student
noise_estimator:  Linear(8->24) -> GELU -> Linear(24->9) -> Sigmoid

Strategy for *bit-exact* FPGA verification:
  1. Quantize to INT8 (symmetric per-tensor weights; binary inputs are exact).
  2. Define a GOLDEN INTEGER model = exactly the integer ops the RTL performs
     (int32 MAC, round-half-up requant via M*x>>S, INT8 activations, 256-entry
      GELU / Sigmoid LUTs).
  3. The RTL is written to mirror those same integer ops, so RTL output ==
     golden-integer output BIT-EXACT by construction (verified in xsim).
  4. Separately report accuracy of the INT8 path vs the float PyTorch output
     (max abs error, and >0.5 binarization agreement) -- this is the real
     "does quantization hurt?" number; literal bit-equality with float is
     impossible under quantization.

Outputs:
  rtl/noise_estimator_params.vh   - weights/biases/LUTs/requant consts (Verilog)
  rtl/tb_vectors.vh               - 20 syndrome inputs + golden int outputs
  golden/golden_report.txt        - accuracy summary
"""
import os, math
import numpy as np
import torch

np.set_printoptions(suppress=True, precision=4)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tools" else HERE
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
GOLDEN_DIR = os.path.join(ROOT, "golden")
RTL  = os.path.join(ROOT, "rtl")
os.makedirs(RTL, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)

S_SHIFT = 16  # requant fixed-point shift, shared by both layers

# ---------------------------------------------------------------- load weights
sd = torch.load(os.path.join(MODEL_DIR, "student_L3_faulty.pt"),
                map_location="cpu", weights_only=False)
W1 = sd["noise_estimator.0.weight"].numpy().astype(np.float64)  # (24,8)
b1 = sd["noise_estimator.0.bias"].numpy().astype(np.float64)    # (24,)
W2 = sd["noise_estimator.2.weight"].numpy().astype(np.float64)  # (9,24)
b2 = sd["noise_estimator.2.bias"].numpy().astype(np.float64)    # (9,)
H, NIN = W1.shape          # 24, 8
NOUT, _ = W2.shape         # 9, 24
print(f"FC1 {W1.shape}  FC2 {W2.shape}")

# ---------------------------------------------------------------- float ref (PyTorch, exact)
def float_noise_est(x):           # x: (B,8) in {0,1}
    xt = torch.tensor(x, dtype=torch.float32)
    y1 = xt @ torch.tensor(W1.T, dtype=torch.float32) + torch.tensor(b1, dtype=torch.float32)
    h  = torch.nn.functional.gelu(y1)               # exact erf GELU (matches nn.GELU())
    y2 = h @ torch.tensor(W2.T, dtype=torch.float32) + torch.tensor(b2, dtype=torch.float32)
    out = torch.sigmoid(y2)
    return y1.numpy(), h.numpy(), y2.numpy(), out.numpy()

# exhaustive calibration over ALL 256 possible 8-bit binary syndromes
allx = np.array([[int(b) for b in f"{v:08b}"] for v in range(256)], dtype=np.float32)
cy1, ch, cy2, cout = float_noise_est(allx)

# ---------------------------------------------------------------- quant scales
scale_w1 = np.abs(W1).max() / 127.0
scale_w2 = np.abs(W2).max() / 127.0
scale_y1 = np.abs(cy1).max() / 127.0    # pre-GELU activation
scale_h  = np.abs(ch).max()  / 127.0    # post-GELU activation (FC2 input)
scale_y2 = np.abs(cy2).max() / 127.0    # pre-sigmoid activation

def qsym(arr, scale):
    return np.clip(np.round(arr / scale), -127, 127).astype(np.int64)

W1q = qsym(W1, scale_w1)                                   # (24,8) int8
W2q = qsym(W2, scale_w2)                                   # (9,24) int8
b1q = np.round(b1 / scale_w1).astype(np.int64)             # bias in acc1 scale
b2q = np.round(b2 / (scale_w2 * scale_h)).astype(np.int64) # bias in acc2 scale

# requant multipliers (M * acc + 2^(S-1)) >> S  ~=  acc * ratio
def make_M(ratio):
    M = int(round(ratio * (1 << S_SHIFT)))
    assert 0 < M < (1 << 31), f"M out of range: {M}"
    return M
M1 = make_M(scale_w1 / scale_y1)                  # acc1(scale_w1) -> y1_q(scale_y1)
M2 = make_M((scale_w2 * scale_h) / scale_y2)      # acc2 -> y2_q(scale_y2)

def requant(acc, M):                              # EXACT op the RTL performs
    t = int(acc) * M + (1 << (S_SHIFT - 1))
    q = t >> S_SHIFT                              # arithmetic floor (matches >>> in HW)
    return max(-128, min(127, q))

# ---------------------------------------------------------------- LUTs (index = q+128, 0..255)
gelu_lut = np.zeros(256, dtype=np.int64)   # int8 out (FC2 input, scale_h)
sig_lut  = np.zeros(256, dtype=np.int64)   # uint8 out: round(sigmoid*255)
for i in range(256):
    q = i - 128
    y1_real = q * scale_y1
    h_real  = float(torch.nn.functional.gelu(torch.tensor(y1_real)).item())
    gelu_lut[i] = int(max(-128, min(127, round(h_real / scale_h))))
    y2_real = q * scale_y2
    sig_lut[i]  = int(max(0, min(255, round(1.0 / (1.0 + math.exp(-y2_real)) * 255.0))))

# ---------------------------------------------------------------- golden INTEGER model
def golden_int(x):                # x: (8,) in {0,1} ints
    x = [int(v) for v in x]
    # FC1
    y1q = np.zeros(H, dtype=np.int64)
    for j in range(H):
        acc = int(b1q[j]) + sum(int(W1q[j, i]) * x[i] for i in range(NIN))
        y1q[j] = requant(acc, M1)
    # GELU LUT
    hq = np.array([gelu_lut[int(y1q[j]) + 128] for j in range(H)], dtype=np.int64)
    # FC2
    y2q = np.zeros(NOUT, dtype=np.int64)
    for k in range(NOUT):
        acc = int(b2q[k]) + sum(int(W2q[k, j]) * int(hq[j]) for j in range(H))
        y2q[k] = requant(acc, M2)
    # Sigmoid LUT  -> uint8 (out_q/255 ~ probability)
    outq = np.array([sig_lut[int(y2q[k]) + 128] for k in range(NOUT)], dtype=np.int64)
    return y2q, outq

# ---------------------------------------------------------------- accuracy vs float (all 256)
out_int_all = np.zeros((256, NOUT))
y2q_all = np.zeros((256, NOUT), dtype=np.int64)
outq_all = np.zeros((256, NOUT), dtype=np.int64)
for v in range(256):
    y2q, outq = golden_int(allx[v].astype(int))
    y2q_all[v] = y2q
    outq_all[v] = outq
    out_int_all[v] = outq / 255.0
max_err_all = np.abs(out_int_all - cout).max()
bin_agree_all = (( (out_int_all > 0.5).astype(int) == (cout > 0.5).astype(int) ).mean())

# ---------------------------------------------------------------- the 20 test vectors
d = np.load(os.path.join(DATA_DIR, "fpga_test_vectors.npz"))
syn20 = d["syndrome"].astype(int)            # (20,8)
# float noise_est golden for the 20 (the npz 'prediction' is the FULL model, not this submodule)
_,_,_, out20_float = float_noise_est(syn20.astype(np.float32))
g_y2q = np.zeros((20, NOUT), dtype=np.int64)
g_outq = np.zeros((20, NOUT), dtype=np.int64)
for i in range(20):
    g_y2q[i], g_outq[i] = golden_int(syn20[i])
out20_int = g_outq / 255.0
max_err20 = np.abs(out20_int - out20_float).max()
bin_agree20 = (((out20_int > 0.5).astype(int) == (out20_float > 0.5).astype(int)).mean())

report = []
def log(s): print(s); report.append(s)
log("="*64)
log("INT8 noise_estimator quantization report")
log("="*64)
log(f"scales: w1={scale_w1:.6g} w2={scale_w2:.6g} y1={scale_y1:.6g} h={scale_h:.6g} y2={scale_y2:.6g}")
log(f"requant: S={S_SHIFT}  M1={M1}  M2={M2}")
log(f"W1q range [{W1q.min()},{W1q.max()}]  W2q range [{W2q.min()},{W2q.max()}]")
log(f"b1q range [{b1q.min()},{b1q.max()}]  b2q range [{b2q.min()},{b2q.max()}]")
log("")
log(f"[ALL 256 syndromes]  max|INT8-float| = {max_err_all:.4f}   >0.5 binarization agreement = {bin_agree_all*100:.2f}%")
log(f"[ 20 test vectors ]  max|INT8-float| = {max_err20:.4f}   >0.5 binarization agreement = {bin_agree20*100:.2f}%")
log("")
log("per-vector (20):  syndrome -> float(>0.5) vs INT8(>0.5)   [match]")
for i in range(20):
    fb = (out20_float[i] > 0.5).astype(int)
    ib = (out20_int[i] > 0.5).astype(int)
    ok = "OK " if (fb==ib).all() else "DIFF"
    log(f"  {i:2d}: {''.join(map(str,syn20[i]))} -> {''.join(map(str,fb))} vs {''.join(map(str,ib))}  [{ok}]")

with open(os.path.join(GOLDEN_DIR, "golden_report.txt"), "w") as f:
    f.write("\n".join(report) + "\n")

# ================================================================ Verilog codegen
def sv_signed_byte(v):   # int8 two's complement -> 2-hex
    return f"{v & 0xFF:02x}"

def sd32(v):             # signed 32-bit decimal literal (sign outside size/base)
    return (f"-32'sd{-int(v)}" if v < 0 else f"32'sd{int(v)}")

lines = []
A = lines.append
A("// AUTO-GENERATED by gen_noise_estimator.py -- do not edit by hand")
A(f"// INT8 noise_estimator params. requant: (M*acc + (1<<{S_SHIFT-1})) >>> {S_SHIFT}")
A(f"localparam integer NIN  = {NIN};")
A(f"localparam integer HID  = {H};")
A(f"localparam integer NOUT = {NOUT};")
A(f"localparam integer SSH  = {S_SHIFT};")
A(f"localparam signed [31:0] M1 = 32'sd{M1};")
A(f"localparam signed [31:0] M2 = 32'sd{M2};")
A("")
# weight / bias / lut ROMs filled in an initial block (synthesizes to ROM)
A("reg signed [7:0]  W1 [0:HID*NIN-1];")
A("reg signed [31:0] B1 [0:HID-1];")
A("reg signed [7:0]  W2 [0:NOUT*HID-1];")
A("reg signed [31:0] B2 [0:NOUT-1];")
A("reg signed [7:0]  GELU_LUT [0:255];")
A("reg [7:0]         SIG_LUT  [0:255];")
A("")
A("initial begin")
for j in range(H):
    for i in range(NIN):
        A(f"  W1[{j*NIN+i}] = 8'sh{sv_signed_byte(int(W1q[j,i]))};")
for j in range(H):
    A(f"  B1[{j}] = {sd32(b1q[j])};")
for k in range(NOUT):
    for j in range(H):
        A(f"  W2[{k*H+j}] = 8'sh{sv_signed_byte(int(W2q[k,j]))};")
for k in range(NOUT):
    A(f"  B2[{k}] = {sd32(b2q[k])};")
for i in range(256):
    A(f"  GELU_LUT[{i}] = 8'sh{sv_signed_byte(int(gelu_lut[i]))};")
for i in range(256):
    A(f"  SIG_LUT[{i}] = 8'h{int(sig_lut[i]) & 0xFF:02x};")
A("end")
with open(os.path.join(RTL, "noise_estimator_params.vh"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote rtl/noise_estimator_params.vh ({len(lines)} lines)")

# ---- testbench vectors: 20 syndromes (8-bit) + golden int outputs (9 bytes each)
tb = []
B = tb.append
B("// AUTO-GENERATED test vectors (20) + golden INTEGER outputs")
B("localparam integer NVEC = 20;")
B("reg [7:0] TB_SYN [0:NVEC-1];          // syndrome, bit i = syn[i]")
B("reg [7:0] TB_OUT [0:NVEC*NOUT-1];     // expected SIG_LUT uint8 output")
B("reg [8:0] TB_BIN [0:NVEC-1];          // expected >0.5 binarization (9 bits)")
B("initial begin")
for i in range(20):
    # pack syndrome bits: bit i of byte = syn20[i][i]
    sb = 0
    for bit in range(8):
        sb |= (int(syn20[i][bit]) & 1) << bit
    B(f"  TB_SYN[{i}] = 8'h{sb:02x};")
for i in range(20):
    for k in range(NOUT):
        B(f"  TB_OUT[{i*NOUT+k}] = 8'h{int(g_outq[i,k]) & 0xFF:02x};")
for i in range(20):
    binv = 0
    for k in range(NOUT):
        binv |= (1 if out20_int[i,k] > 0.5 else 0) << k
    B(f"  TB_BIN[{i}] = 9'h{binv:03x};")
B("end")
with open(os.path.join(RTL, "tb_vectors.vh"), "w") as f:
    f.write("\n".join(tb) + "\n")
print("wrote rtl/tb_vectors.vh")
print("DONE")
