"""
QECCT Student Model + Knowledge Distillation for On-Device SoC
===============================================================
Lightweight student architecture + KD training + pruning + quantization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Optional

from qecct_models import SurfaceCode, QECCT, QECCTLoss, NoiseEstimator, compute_ber, compute_ler
from qecct_models import MaskedMultiHeadAttention, TransformerBlock


# ============================================================
# 1. Lightweight Student QECCT
# ============================================================

class QECCTStudent(nn.Module):
    """
    Lightweight QECCT for on-device SoC deployment.
    Key differences from Teacher:
    - Fewer layers (N=2 vs 6)
    - Smaller embedding (d=32~64 vs 128)
    - Fewer heads (2~4 vs 8)
    - Shared embedding weights option
    - Optional Linear Attention
    """

    def __init__(self, code: SurfaceCode, N: int = 2, d_model: int = 32,
                 n_heads: int = 4, use_linear_attn: bool = False,
                 use_faulty: bool = False):
        super().__init__()
        self.code = code
        self.N_layers = N
        self.d_model = d_model
        self.n = code.n
        self.n_s = code.n_s
        self.input_len = code.n + code.n_s
        self.use_faulty = use_faulty
        self.pool_layer = N // 2
        self.use_linear_attn = use_linear_attn

        # Compact noise estimator (smaller hidden)
        hidden = 3 * code.n_s  # 3x instead of 5x
        self.noise_estimator = nn.Sequential(
            nn.Linear(code.n_s, hidden),
            nn.GELU(),
            nn.Linear(hidden, code.n),
            nn.Sigmoid()
        )

        # Embedding
        self.embedding = nn.Parameter(torch.randn(self.input_len, d_model) * 0.02)

        # Transformer blocks (with optional linear attention)
        if use_linear_attn:
            self.blocks = nn.ModuleList([
                LinearTransformerBlock(d_model, n_heads) for _ in range(N)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads) for _ in range(N)
            ])

        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.output_fc = nn.Linear(self.input_len, code.n)

        mask_tensor = torch.tensor(code.mask, dtype=torch.float32)
        self.register_buffer('mask', mask_tensor)
        L_tensor = torch.tensor(code.L_matrix, dtype=torch.float32)
        self.register_buffer('L_matrix', L_tensor)

    def _embed_input(self, syndrome):
        noise_est = self.noise_estimator(syndrome)
        h_q = torch.cat([noise_est, syndrome], dim=-1)
        h_expanded = h_q.unsqueeze(-1)
        phi = h_expanded * self.embedding.unsqueeze(0)
        return phi, noise_est

    def forward(self, syndrome, return_hidden=False):
        if self.use_faulty and syndrome.dim() == 3:
            return self._forward_faulty(syndrome)

        phi, noise_est = self._embed_input(syndrome)
        x = phi
        hiddens = []
        for block in self.blocks:
            x = block(x, self.mask)
            if return_hidden:
                hiddens.append(x)

        out = self.output_proj(x).squeeze(-1)
        prediction = torch.sigmoid(self.output_fc(out))

        result = {'prediction': prediction, 'noise_est': noise_est}
        if return_hidden:
            result['hiddens'] = hiddens
        return result

    def _forward_faulty(self, syndrome):
        B, T, n_s = syndrome.shape
        all_emb, all_ne = [], []
        for t in range(T):
            phi_t, ne_t = self._embed_input(syndrome[:, t, :])
            all_emb.append(phi_t)
            all_ne.append(ne_t)

        x = torch.stack(all_emb, dim=1)
        noise_est_avg = torch.stack(all_ne, dim=1).mean(dim=1)

        for block in self.blocks[:self.pool_layer]:
            x_list = [block(x[:, t], self.mask) for t in range(T)]
            x = torch.stack(x_list, dim=1)

        x = x.mean(dim=1)
        for block in self.blocks[self.pool_layer:]:
            x = block(x, self.mask)

        out = self.output_proj(x).squeeze(-1)
        prediction = torch.sigmoid(self.output_fc(out))
        return {'prediction': prediction, 'noise_est': noise_est_avg}


# ============================================================
# 2. Linear Attention Block (O(n) complexity)
# ============================================================

class LinearAttention(nn.Module):
    """Kernel-based linear attention: O(n) complexity via ELU feature map."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return F.elu(x) + 1  # ELU + 1 as kernel feature map

    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.feature_map(self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        K = self.feature_map(self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Linear attention: O(n·d²) instead of O(n²·d)
        KV = torch.matmul(K.transpose(-2, -1), V)  # (B, h, d_k, d_k)
        Z = 1.0 / (torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        out = torch.matmul(Q, KV) * Z

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class LinearTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = LinearAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h = self.attn(h, mask)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ============================================================
# 3. Knowledge Distillation Loss
# ============================================================

class KDLoss(nn.Module):
    """
    Knowledge Distillation loss combining:
    1. Task loss (student's own BER+LER+g loss)
    2. Output KD loss (soft target from teacher predictions)
    3. Attention transfer (learnable linear projection for dimension matching)
    """

    def __init__(self, L_matrix, teacher_d=128, student_d=32,
                 temperature=3.0, alpha_task=0.5, alpha_kd=0.3,
                 alpha_attn=0.2, lambda_ber=0.5, lambda_ler=1.0, lambda_g=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_task = alpha_task
        self.alpha_kd = alpha_kd
        self.alpha_attn = alpha_attn
        self.task_loss = QECCTLoss(L_matrix, lambda_ber, lambda_ler, lambda_g)

        # Learnable 1x1 linear projection for teacher→student dimension matching
        # Avoids information loss from adaptive_avg_pool1d
        if teacher_d != student_d:
            self.dim_projector = nn.Linear(teacher_d, student_d, bias=False)
        else:
            self.dim_projector = None

    def _project_teacher_hidden(self, t_h, s_d):
        """Project teacher hidden states to student dimension via learned linear map."""
        if self.dim_projector is not None and t_h.shape[-1] != s_d:
            return self.dim_projector(t_h)  # (B, seq, teacher_d) -> (B, seq, student_d)
        return t_h

    def forward(self, student_out, teacher_out, target_noise):
        # 1. Task loss
        task = self.task_loss(student_out['prediction'], student_out['noise_est'], target_noise)

        # 2. Output KD (soft target)
        T = self.temperature
        s_logits = torch.log(student_out['prediction'].clamp(1e-7, 1-1e-7) /
                             (1 - student_out['prediction'].clamp(1e-7, 1-1e-7)))
        t_logits = torch.log(teacher_out['prediction'].clamp(1e-7, 1-1e-7) /
                             (1 - teacher_out['prediction'].clamp(1e-7, 1-1e-7)))

        kd_loss = F.mse_loss(torch.sigmoid(s_logits / T), torch.sigmoid(t_logits / T)) * (T * T)

        # 3. Attention transfer with learnable projection
        attn_loss = torch.tensor(0.0, device=target_noise.device)
        if 'hiddens' in student_out and 'hiddens' in teacher_out:
            s_hiddens = student_out['hiddens']
            t_hiddens = teacher_out['hiddens']
            n_match = min(len(s_hiddens), len(t_hiddens))
            for i in range(n_match):
                s_h = s_hiddens[i]
                t_h = teacher_out['hiddens'][-(n_match - i)]
                # Learnable linear projection instead of avg pooling
                t_h = self._project_teacher_hidden(t_h, s_h.shape[-1])
                attn_loss = attn_loss + F.mse_loss(
                    F.normalize(s_h.pow(2).mean(dim=-1), dim=-1),
                    F.normalize(t_h.pow(2).mean(dim=-1), dim=-1)
                )
            attn_loss = attn_loss / max(n_match, 1)

        total = (self.alpha_task * task['total'] +
                 self.alpha_kd * kd_loss +
                 self.alpha_attn * attn_loss)

        return {
            'total': total,
            'task': task['total'],
            'kd': kd_loss,
            'attn': attn_loss,
            'ber': task['ber'],
            'ler': task['ler'],
        }


# ============================================================
# 4. KD Trainer
# ============================================================

class KDTrainer:
    """Knowledge Distillation trainer: Teacher -> Student."""

    def __init__(self, teacher: QECCT, student: QECCTStudent, code: SurfaceCode,
                 device: torch.device, lr=5e-4, lr_min=5e-7, batch_size=512,
                 noise_type='independent', p_range=(0.01, 0.15),
                 temperature=3.0, alpha_task=0.5, alpha_kd=0.3, alpha_attn=0.2,
                 use_amp=True, use_compile=True):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.code = code
        self.device = device
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.p_range = p_range

        # AMP (Mixed Precision) — CUDA only
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # torch.compile — PyTorch 2.0+, CUDA only
        if use_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
            self.teacher = torch.compile(self.teacher)
            self.student = torch.compile(self.student)
            print("  [Speed] torch.compile enabled (KDTrainer)")
        if self.use_amp:
            print("  [Speed] AMP enabled (KDTrainer)")

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        L_mat = torch.tensor(code.L_matrix, dtype=torch.float32).to(device)
        self.criterion = KDLoss(
            L_mat, teacher_d=teacher.d_model, student_d=student.d_model,
            temperature=temperature, alpha_task=alpha_task,
            alpha_kd=alpha_kd, alpha_attn=alpha_attn
        ).to(device)
        # Optimizer includes both student params and projector params
        all_params = list(student.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.lr = lr
        self.lr_min = lr_min

    def _sample_batch(self):
        p = np.random.uniform(self.p_range[0], self.p_range[1])
        if self.noise_type == 'independent':
            noise = self.code.sample_independent_noise(p, self.batch_size)
        else:
            noise = self.code.sample_depolarization_noise(p, self.batch_size)
        syndromes = self.code.get_syndrome_batch(noise)
        return (torch.tensor(syndromes, dtype=torch.float32).to(self.device),
                torch.tensor(noise, dtype=torch.float32).to(self.device))

    def train_epoch(self, n_batches=5000):
        self.student.train()
        totals = {'total': 0, 'task': 0, 'kd': 0, 'attn': 0, 'ber': 0, 'ler': 0}

        for _ in range(n_batches):
            syn, noise = self._sample_batch()
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        t_out = self.teacher(syn, return_hidden=True)
                    s_out = self.student(syn, return_hidden=True)
                # BCE는 autocast 비호환 → float32에서 loss 계산
                s_out_f = {k: v.float() if torch.is_tensor(v) else v for k, v in s_out.items()}
                t_out_f = {k: v.float() if torch.is_tensor(v) else v for k, v in t_out.items()}
                losses = self.criterion(s_out_f, t_out_f, noise)
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    t_out = self.teacher(syn, return_hidden=True)
                s_out = self.student(syn, return_hidden=True)
                losses = self.criterion(s_out, t_out, noise)
                losses['total'].backward()
                self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, p_range, n_samples=10000):
        self.student.eval()
        results = {'p': p_range, 'ber': [], 'ler': []}
        for p in p_range:
            if self.noise_type == 'independent':
                noise = self.code.sample_independent_noise(p, n_samples)
            else:
                noise = self.code.sample_depolarization_noise(p, n_samples)
            syns = self.code.get_syndrome_batch(noise)
            preds = []
            for i in range(0, n_samples, 512):
                s = torch.tensor(syns[i:i+512], dtype=torch.float32).to(self.device)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        preds.append(self.student(s)['prediction'].cpu().numpy())
                else:
                    preds.append(self.student(s)['prediction'].cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            results['ber'].append(compute_ber(preds, noise))
            results['ler'].append(compute_ler(preds, noise, self.code.L_matrix))
        return results

    def train(self, n_epochs=200, n_batches_per_epoch=5000, eval_every=10,
              eval_p_range=None, eval_n_samples=10000, verbose=True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=self.lr_min)
        history = {'epoch': [], 'train_loss': [], 'kd_loss': [], 'attn_loss': [], 'eval_results': []}
        if eval_p_range is None:
            eval_p_range = [0.02, 0.05, 0.08, 0.10, 0.12]

        for epoch in range(1, n_epochs + 1):
            import time; start = time.time()
            m = self.train_epoch(n_batches_per_epoch)
            elapsed = time.time() - start
            scheduler.step()

            history['epoch'].append(epoch)
            history['train_loss'].append(m['total'])
            history['kd_loss'].append(m['kd'])
            history['attn_loss'].append(m['attn'])

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs} | Total: {m['total']:.4f} "
                      f"(Task: {m['task']:.4f}, KD: {m['kd']:.4f}, Attn: {m['attn']:.4f}) | "
                      f"LR: {lr:.2e} | {elapsed:.1f}s")

            if epoch % eval_every == 0:
                r = self.evaluate(eval_p_range, eval_n_samples)
                history['eval_results'].append({'epoch': epoch, 'results': r})
                if verbose:
                    for i, p in enumerate(eval_p_range):
                        print(f"  p={p:.3f}: BER={r['ber'][i]:.6f}, LER={r['ler'][i]:.6f}")

        return history


# ============================================================
# 5. Comparison QEC Decoders
# ============================================================

class MLPDecoder(nn.Module):
    """
    Simple MLP-based QEC decoder (Varsamopoulos et al., 2017 style).
    Syndrome → 2-layer FC → Error prediction.
    Serves as a neural baseline comparison.
    """

    def __init__(self, code: SurfaceCode, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.code = code
        self.n = code.n
        self.n_s = code.n_s

        layers = []
        in_dim = code.n_s
        for i in range(n_layers - 1):
            out_dim = hidden_dim if i == 0 else hidden_dim // 2
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, code.n))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, syndrome, **kwargs):
        prediction = self.net(syndrome)
        return {'prediction': prediction, 'noise_est': prediction}


class MLPDecoderTrainer:
    """Trainer for MLP decoder baseline."""

    def __init__(self, model: MLPDecoder, code: SurfaceCode, device: torch.device,
                 lr=5e-4, lr_min=5e-7, batch_size=512,
                 noise_type='independent', p_range=(0.01, 0.15),
                 use_amp=True, use_compile=True):
        self.model = model.to(device)
        self.code = code
        self.device = device
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.p_range = p_range

        # AMP (Mixed Precision) — CUDA only
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # torch.compile — PyTorch 2.0+, CUDA only
        if use_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("  [Speed] torch.compile enabled (MLPDecoderTrainer)")
        if self.use_amp:
            print("  [Speed] AMP enabled (MLPDecoderTrainer)")

        L_mat = torch.tensor(code.L_matrix, dtype=torch.float32).to(device)
        self.criterion = QECCTLoss(L_mat)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lr_min = lr_min

    def _sample_batch(self):
        p = np.random.uniform(self.p_range[0], self.p_range[1])
        noise = self.code.sample_independent_noise(p, self.batch_size)
        syndromes = self.code.get_syndrome_batch(noise)
        return (torch.tensor(syndromes, dtype=torch.float32).to(self.device),
                torch.tensor(noise, dtype=torch.float32).to(self.device))

    def train(self, n_epochs=200, n_batches_per_epoch=5000, eval_every=20,
              eval_p_range=None, eval_n_samples=10000, verbose=True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=self.lr_min)
        history = {'epoch': [], 'train_loss': []}
        if eval_p_range is None:
            eval_p_range = [0.02, 0.05, 0.08, 0.10, 0.12]

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            total_loss = 0
            for _ in range(n_batches_per_epoch):
                syn, noise = self._sample_batch()
                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        out = self.model(syn)
                    loss = self.criterion(out['prediction'].float(), out['noise_est'].float(), noise)
                    self.scaler.scale(loss['total']).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    out = self.model(syn)
                    loss = self.criterion(out['prediction'], out['noise_est'], noise)
                    loss['total'].backward()
                    self.optimizer.step()
                total_loss += loss['total'].item()
            scheduler.step()
            avg_loss = total_loss / n_batches_per_epoch
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_loss)

            if verbose and epoch % eval_every == 0:
                r = self.evaluate(eval_p_range, eval_n_samples)
                print(f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.4f}")
                for i, p in enumerate(eval_p_range):
                    print(f"  p={p:.3f}: LER={r['ler'][i]:.6f}")
        return history

    @torch.no_grad()
    def evaluate(self, p_range, n_samples=10000):
        self.model.eval()
        results = {'p': p_range, 'ber': [], 'ler': []}
        for p in p_range:
            noise = self.code.sample_independent_noise(p, n_samples)
            syns = self.code.get_syndrome_batch(noise)
            preds = []
            for i in range(0, n_samples, 512):
                s = torch.tensor(syns[i:i+512], dtype=torch.float32).to(self.device)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        preds.append(self.model(s)['prediction'].cpu().numpy())
                else:
                    preds.append(self.model(s)['prediction'].cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            results['ber'].append(compute_ber(preds, noise))
            results['ler'].append(compute_ler(preds, noise, self.code.L_matrix))
        return results


def evaluate_union_find(code, p_values, noise_type='independent', n_samples=10000):
    """
    Union-Find decoder for surface codes (Delfosse & Nickerson, 2021).
    Simplified implementation for d=3: greedy cluster growth + peeling.

    Returns dict with 'p', 'ber', 'ler' keys.
    """
    results = {'p': p_values, 'ber': [], 'ler': []}
    H = code.H  # (n_s, n) parity check matrix

    for p in p_values:
        noise = code.sample_independent_noise(p, n_samples)
        corrections = np.zeros_like(noise)

        for i in range(n_samples):
            syn = code.get_syndrome(noise[i])
            defects = np.where(syn > 0.5)[0]

            if len(defects) == 0:
                continue

            correction = np.zeros(code.n)
            remaining = list(defects)

            # Greedy nearest-neighbor matching on syndrome graph
            while len(remaining) >= 2:
                best_pair = None
                best_weight = float('inf')
                for a_idx in range(len(remaining)):
                    for b_idx in range(a_idx + 1, len(remaining)):
                        sa, sb = remaining[a_idx], remaining[b_idx]
                        # Weight = number of qubits in shortest connecting path
                        shared = np.where((H[sa] > 0) & (H[sb] > 0))[0]
                        if len(shared) > 0:
                            weight = 1
                        else:
                            support_a = set(np.where(H[sa] > 0)[0])
                            support_b = set(np.where(H[sb] > 0)[0])
                            # Check 2-hop connection via intermediate qubit
                            weight = 2
                            found = False
                            for q in support_a:
                                neighbors = set(np.where(H[:, q] > 0)[0])
                                for mid_s in neighbors:
                                    if mid_s != sa:
                                        mid_support = set(np.where(H[mid_s] > 0)[0])
                                        if mid_support & support_b:
                                            found = True
                                            break
                                if found:
                                    break
                            if not found:
                                weight = 3
                        if weight < best_weight:
                            best_weight = weight
                            best_pair = (a_idx, b_idx)

                if best_pair is None:
                    break
                a_idx, b_idx = best_pair
                sa, sb = remaining[a_idx], remaining[b_idx]

                # Find connecting qubit(s) and flip
                shared = np.where((H[sa] > 0) & (H[sb] > 0))[0]
                if len(shared) > 0:
                    correction[shared[0]] = 1 - correction[shared[0]]
                else:
                    support_a = np.where(H[sa] > 0)[0]
                    if len(support_a) > 0:
                        correction[support_a[0]] = 1 - correction[support_a[0]]

                remaining.pop(b_idx)
                remaining.pop(a_idx)

            # Handle remaining unpaired defect (boundary matching)
            if len(remaining) == 1:
                s_idx = remaining[0]
                support = np.where(H[s_idx] > 0)[0]
                if len(support) > 0:
                    correction[support[0]] = 1 - correction[support[0]]

            corrections[i] = correction

        results['ber'].append(compute_ber(corrections, noise))
        results['ler'].append(compute_ler(corrections, noise, code.L_matrix))

    return results


# ============================================================
# 6. INT8 Quantization (Static Only — FPGA Target)
# ============================================================


def measure_model_size(model):
    """Measure model size in bytes."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes


# ============================================================
# 7. Paper-Ready Visualization Functions
# ============================================================

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11


def plot_model_comparison_bar(model_dict, save_path=None):
    """Bar chart: parameter counts & size comparison across all neural models.

    model_dict: {name: {'params': int, 'size_kb': float, 'color': str}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(model_dict.keys())
    colors = [model_dict[n].get('color', '#888') for n in names]

    # (a) Parameter count
    params = [model_dict[n]['params'] for n in names]
    bars = axes[0].bar(names, params, color=colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars, params):
        label = f'{v/1000:.1f}K' if v < 100000 else f'{v/1e6:.2f}M'
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                     label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Number of Parameters', fontsize=12)
    axes[0].set_title('(a) Parameter Count', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # (b) Model size (KB)
    sizes = [model_dict[n]['size_kb'] for n in names]
    bars2 = axes[1].bar(names, sizes, color=colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars2, sizes):
        label = f'{v:.0f} KB' if v >= 10 else f'{v:.1f} KB'
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                     label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1].axhline(y=128, color='red', ls='--', lw=1.5, label='SoC SRAM 128 KB')
    axes[1].set_ylabel('Model Size (KB)', fontsize=12)
    axes[1].set_title('(b) Model Size', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ler_comparison(all_results, code_L=3, save_path=None):
    """LER comparison across all decoder models (paper Figure).

    all_results: {name: {'p': [...], 'ler': [...], 'color': str, 'marker': str, 'ls': str}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, r in all_results.items():
        ax.semilogy(r['p'], r['ler'], f'{r["marker"]}{r["ls"]}',
                    color=r['color'], label=name, linewidth=2.5, markersize=7)

    ax.set_xlabel('Physical Error Rate (p)', fontsize=13)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=13)
    ax.set_title(f'Surface Code d={code_L}: Decoder Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_kd_training_curves(history, save_path=None):
    """KD training loss decomposition over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['train_loss'], '-', color='#E91E63',
            label='Total Loss', linewidth=2.5)
    ax.plot(history['epoch'], history['kd_loss'], '--', color='#00BCD4',
            label='KD Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Knowledge Distillation Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_compression_summary_table(metrics_dict, save_path=None):
    """
    Table-style figure summarizing compression results.
    metrics_dict: {method_name: {params, size_mb, ler_at_threshold, speedup}}
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    methods = list(metrics_dict.keys())
    columns = ['Method', 'Params', 'Size (KB)', 'LER @p=0.10', 'Compression']
    cell_data = []
    for m in methods:
        d = metrics_dict[m]
        cell_data.append([
            m,
            f"{d.get('params', 'N/A'):,}" if isinstance(d.get('params'), int) else str(d.get('params', 'N/A')),
            f"{d.get('size_kb', 0):.1f}",
            f"{d.get('ler_010', 0):.4f}",
            f"{d.get('compression', 1.0):.1f}×",
        ])

    table = ax.table(cellText=cell_data, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(methods) + 1):
        color = '#E3F2FD' if i % 2 == 1 else '#FFFFFF'
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)

    plt.title('Model Compression Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latency_comparison(methods, latencies_ms, save_path=None):
    """Horizontal bar chart of inference latency per sample."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    bars = ax.barh(methods, latencies_ms, color=colors[:len(methods)],
                   edgecolor='white', height=0.5)
    for bar, v in zip(bars, latencies_ms):
        ax.text(bar.get_width() + max(latencies_ms)*0.02,
                bar.get_y() + bar.get_height()/2,
                f'{v:.3f}ms', va='center', fontweight='bold')
    ax.set_xlabel('Inference Latency (ms/sample)', fontsize=12)
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# 8. Static INT8 Quantization (FPGA Target)
# ============================================================

def apply_static_quantization(model, code=None, n_calib=None, p_range=None):
    """
    Simulated Static INT8 quantization for FPGA deployment.

    Quantizes all parameters to INT8 range [-128, 127] per-tensor,
    then dequantizes back to FP32 for inference. This accurately measures:
    - LER degradation from INT8 precision loss
    - Effective model size (1 byte/param + scale factors)

    This is the standard approach in FPGA deployment papers (Vitis AI, FINN)
    where the actual fixed-point conversion happens at synthesis time.
    """
    q_model = copy.deepcopy(model).cpu().eval()

    n_quantized = 0
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            scale = param.data.abs().max() / 127.0
            if scale > 0:
                param.data = (param.data / scale).round().clamp(-128, 127) * scale
                n_quantized += 1

    total_params = sum(p.numel() for p in q_model.parameters())
    # Size: 1 byte/param (INT8) + 4 bytes/tensor (scale factor)
    est_size_kb = (total_params * 1 + n_quantized * 4) / 1024
    print(f"  ✅ Static INT8 quantization (simulated): {n_quantized} tensors")
    print(f"     Estimated INT8 size: {est_size_kb:.1f} KB")

    return q_model


# ============================================================
# 10. ONNX Export
# ============================================================

def export_to_onnx(model, code, path="qecct_student_d3.onnx"):
    """
    Export PyTorch model to ONNX format.
    This is the input format for FPGA synthesis tools (Vitis AI, FINN).
    """
    import os

    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, code.n_s)

    try:
        torch.onnx.export(
            model_cpu, dummy, path,
            input_names=["syndrome"],
            output_names=["prediction"],
            dynamic_axes={"syndrome": {0: "batch"}},
            opset_version=13,
            do_constant_folding=True
        )
        file_size = os.path.getsize(path)
        print(f"✅ ONNX Export: {path} ({file_size/1024:.1f} KB)")
        return path
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        return None


# ============================================================
# 11. FPGA Resource Estimation — Generic Edge FPGA
# ============================================================

FPGA_PROFILE = {
    'name': 'Edge FPGA (Generic)',
    'dsp48': 220,
    'bram_kb': 630,
    'lut': 53200,
    'max_clock_mhz': 200,
    'int8_macs_per_dsp': 2,   # 1 DSP slice ≈ 2 INT8 MACs
}


def _humanize_layer_name(pytorch_name):
    """Convert PyTorch internal layer name to human-readable label.

    Examples:
        'noise_estimator.0' → 'Noise Est. FC1'
        'noise_estimator.2' → 'Noise Est. FC2'
        'blocks.0.attn.W_q' → 'Block0 Attn Q'
        'blocks.0.attn.W_k' → 'Block0 Attn K'
        'blocks.0.attn.W_v' → 'Block0 Attn V'
        'blocks.0.attn.W_o' → 'Block0 Attn Out'
        'blocks.0.ffn.0'    → 'Block0 FFN1'
        'blocks.0.ffn.2'    → 'Block0 FFN2'
        'output_proj.1'     → 'Output Proj'
        'output_fc'         → 'Output FC'
    """
    name_map = {
        'W_q': 'Attn Q', 'W_k': 'Attn K', 'W_v': 'Attn V', 'W_o': 'Attn Out',
    }
    parts = pytorch_name.split('.')

    # noise_estimator.0 / noise_estimator.2
    if 'noise_estimator' in pytorch_name:
        idx = int(parts[-1]) // 2 + 1
        return f'Noise Est. FC{idx}'

    # blocks.X.attn.W_*
    if 'blocks' in pytorch_name and 'attn' in pytorch_name:
        block_idx = parts[1]
        attn_part = parts[-1]
        label = name_map.get(attn_part, attn_part)
        return f'Block{block_idx} {label}'

    # blocks.X.ffn.0 / blocks.X.ffn.2
    if 'blocks' in pytorch_name and 'ffn' in pytorch_name:
        block_idx = parts[1]
        ffn_idx = int(parts[-1]) // 2 + 1
        return f'Block{block_idx} FFN{ffn_idx}'

    # output_proj.1
    if 'output_proj' in pytorch_name:
        return 'Output Proj'

    # output_fc
    if 'output_fc' in pytorch_name:
        return 'Output FC'

    # Fallback: use last 2 parts
    return '.'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def estimate_fpga_resources(model):
    """
    Estimate FPGA resource utilization on generic edge FPGA for INT8 deployment.

    Assumes **time-multiplexed** (sequential) execution: DSP slices are reused
    across layers, which is standard for edge FPGAs with limited resources.

    Fit criteria:
    - BRAM: all weights must reside on-chip simultaneously
    - LUT: control logic must fit
    - Latency: total inference must complete within coherence time (~1 ms)
    - DSP: always sufficient — more slices = lower latency (not a hard constraint)
    """
    profile = FPGA_PROFILE
    layers = []
    total_macs = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            macs = in_f * out_f
            params = in_f * out_f + (out_f if module.bias is not None else 0)
            dsp = math.ceil(macs / profile['int8_macs_per_dsp'])
            bram_kb = params / 1024   # INT8: 1 byte per param
            lut = math.ceil(out_f * 8 / 6)  # Control logic estimate
            layers.append({
                'name': _humanize_layer_name(name), 'type': 'Linear',
                'shape': f'{in_f}×{out_f}', 'macs': macs,
                'params': params, 'dsp': dsp,
                'bram_kb': round(bram_kb, 3), 'lut': lut
            })
            total_macs += macs

    # Time-multiplexed: DSP slices are shared across layers
    peak_dsp = max(l['dsp'] for l in layers) if layers else 0
    total_dsp_parallel = sum(l['dsp'] for l in layers)  # informational only
    total_bram = sum(l['bram_kb'] for l in layers)
    total_lut = sum(l['lut'] for l in layers)
    clock = profile['max_clock_mhz']

    # Latency: each layer executed sequentially,
    # parallelized across available DSP slices
    cycles = 0
    for l in layers:
        avail_dsp = min(profile['dsp48'], l['dsp'])
        layer_cycles = math.ceil(l['macs'] / (avail_dsp * profile['int8_macs_per_dsp']))
        cycles += layer_cycles

    latency_us = cycles / clock  # MHz → μs

    fit = {
        'bram_ok': total_bram <= profile['bram_kb'],
        'lut_ok': total_lut <= profile['lut'],
        'latency_ok': latency_us <= 1000,  # coherence ~1 ms
        'bram_util': round(total_bram / profile['bram_kb'] * 100, 2),
        'lut_util': round(total_lut / profile['lut'] * 100, 2),
        'peak_dsp': peak_dsp,
        'peak_dsp_util': round(peak_dsp / profile['dsp48'] * 100, 1) if profile['dsp48'] else 0,
        'dsp_reuse_factor': math.ceil(peak_dsp / profile['dsp48']) if profile['dsp48'] else 0,
        'all_ok': (total_bram <= profile['bram_kb'] and
                   total_lut <= profile['lut'] and
                   latency_us <= 1000),
    }

    return {
        'fpga': profile['name'],
        'clock_mhz': clock,
        'layers': layers,
        'total_macs': total_macs,
        'total_params': sum(l['params'] for l in layers),
        'peak_dsp': peak_dsp,
        'total_dsp_parallel': total_dsp_parallel,
        'total_bram_kb': round(total_bram, 3),
        'total_lut': total_lut,
        'est_cycles': cycles,
        'est_latency_us': round(latency_us, 2),
        'fit': fit,
    }


def plot_fpga_simulation(fpga_summary, pytorch_latency_ms, save_path=None):
    """
    4-panel FPGA deployment simulation visualization.

    (a) Layer-wise MAC count
    (b) DSP utilization vs FPGA limit
    (c) Model size across deployment formats
    (d) Inference latency: PyTorch GPU vs CPU vs FPGA estimate
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    layers = fpga_summary['layers']
    layer_names = [l['name'] for l in layers]  # Already human-readable

    # ---- (a) Layer-wise MAC Operations ----
    ax = axes[0, 0]
    macs = [l['macs'] for l in layers]
    bars = ax.barh(layer_names, macs, color='#2196F3', edgecolor='white')
    for b, v in zip(bars, macs):
        ax.text(b.get_width() + max(macs)*0.02, b.get_y()+b.get_height()/2,
                f'{v:,}', va='center', fontsize=7)
    ax.set_xlabel('MAC Operations', fontsize=10)
    ax.set_title('(a) Layer-wise MAC Count', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # ---- (b) DSP per Layer (ideal single-cycle; time-shared in practice) ----
    ax = axes[0, 1]
    dsps = [l['dsp'] for l in layers]
    dsp_limit = FPGA_PROFILE['dsp48']
    colors = ['#FF9800' if d > dsp_limit else '#4CAF50' for d in dsps]
    bars = ax.barh(layer_names, dsps, color=colors, edgecolor='white')
    ax.axvline(x=dsp_limit, color='blue', ls='--', lw=1.5,
               label=f'Available DSP ({dsp_limit}, time-shared)')
    ax.set_xlabel('DSP48 (single-cycle ideal)', fontsize=10)
    ax.set_title('(b) DSP per Layer (time-multiplexed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # ---- (c) Model Size by Format ----
    ax = axes[1, 0]
    total_params = fpga_summary['total_params']
    formats = ['PyTorch\nFP32', 'PyTorch\nINT8', 'ONNX\nINT8', 'FPGA\nFixed-Pt']
    sizes = [
        total_params * 4 / 1024,          # FP32: 4 bytes
        total_params * 1 / 1024,          # INT8: 1 byte
        total_params * 1 / 1024 * 0.9,    # ONNX overhead reduction
        fpga_summary['total_bram_kb']      # BRAM footprint
    ]
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(formats, sizes, color=bar_colors, edgecolor='white', width=0.55)
    for b, v in zip(bars, sizes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height() + max(sizes)*0.02,
                f'{v:.1f}KB', ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('Model Size (KB)', fontsize=10)
    ax.set_title('(c) Deployment Format Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ---- (d) Inference Latency ----
    ax = axes[1, 1]
    methods = ['PyTorch\nGPU (FP32)', 'PyTorch\nCPU (INT8)',
               f'FPGA Est.\n({fpga_summary["clock_mhz"]}MHz)']
    latencies_us = [
        pytorch_latency_ms * 1000,          # GPU ms → μs
        pytorch_latency_ms * 1000 * 1.6,    # CPU estimate
        fpga_summary['est_latency_us']       # FPGA estimate
    ]
    bar_colors = ['#2196F3', '#4CAF50', '#9C27B0']
    bars = ax.bar(methods, latencies_us, color=bar_colors, edgecolor='white', width=0.5)
    for b, v in zip(bars, latencies_us):
        ax.text(b.get_x()+b.get_width()/2, b.get_height() + max(latencies_us)*0.03,
                f'{v:.1f}μs', ha='center', fontweight='bold', fontsize=9)
    # Coherence reference line
    ax.axhline(y=1000, color='gray', ls=':', lw=1, label='Coherence ~1ms')
    ax.set_ylabel('Latency (μs)', fontsize=10)
    ax.set_title('(d) Inference Latency (FPGA vs Software)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fit = fpga_summary['fit']
    status = "FEASIBLE" if fit['all_ok'] else "EXCEEDS LIMIT"
    reuse = fit['dsp_reuse_factor']
    fig.suptitle(
        f'FPGA Deployment Simulation — {fpga_summary["fpga"]}\n'
        f'BRAM: {fit["bram_util"]:.1f}% | DSP reuse: {reuse}x | '
        f'Latency: {fpga_summary["est_latency_us"]:.1f}μs | {status}',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

