# models/cpc_module.py
from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Any

class CPCModule(nn.Module):
    encoder: nn.Module
    projection_dim: int
    gru_hidden_dim: int
    num_future_steps: int
    temperature: float = 0.1

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        # x_seq: [B, T, *obs_shape]
        B, T = x_seq.shape[:2]
        z_seq = jax.vmap(jax.vmap(self.encoder))(x_seq)  # [B, T, D]
        z_proj = nn.Dense(self.projection_dim)(z_seq)

        # GRU context model
        gru = nn.GRUCell()
        carry = gru.initialize_carry(jax.random.PRNGKey(0), (B,), self.gru_hidden_dim)
        context_seq = []
        for t in range(T):
            carry, out = gru(carry, z_proj[:, t])
            context_seq.append(out)
        c_seq = jnp.stack(context_seq, axis=1)

        # InfoNCE loss
        loss = 0.0
        for k in range(1, self.num_future_steps + 1):
            if T - k <= 0:
                continue
            z_pos = z_proj[:, k:]      # [B, T-k, D]
            c_t = c_seq[:, :-k]        # [B, T-k, H]
            pred = nn.Dense(self.projection_dim, name=f"pred_{k}")(c_t)
            logits = jnp.einsum("btd,bkd->btk", pred, z_pos) / self.temperature
            labels = jnp.arange(logits.shape[1])
            loss += jnp.mean(-jax.nn.log_softmax(logits, axis=-1)[..., labels])
        return loss / self.num_future_steps
