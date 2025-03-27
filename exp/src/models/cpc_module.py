from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Any


class CPCModule(nn.Module):
    encoder: nn.Module
    projection_dim: int
    gru_hidden_dim: int
    num_future_steps: int
    temperature: float = 0.1

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        # x_seq: [B, T, H, W, C] or [B, T, obs_dim]
        B, T = x_seq.shape[:2]

        # z_t = encoder(x_t)
        z_seq = jax.vmap(jax.vmap(self.encoder))(x_seq)  # [B, T, D]
        z_seq = nn.Dense(self.projection_dim)(z_seq)

        # GRU to produce context c_t
        gru = nn.GRUCell()
        carry = gru.initialize_carry(jax.random.PRNGKey(0), (B,), self.gru_hidden_dim)

        c_seq = []
        for t in range(T):
            carry, out = gru(carry, z_seq[:, t])
            c_seq.append(out)
        c_seq = jnp.stack(c_seq, axis=1)  # [B, T, H]

        # Contrastive predictions and InfoNCE loss
        loss = 0.0
        for k in range(1, self.num_future_steps + 1):
            if T - k <= 0:
                continue
            z_pos = z_seq[:, k:]      # [B, T-k, D]
            c_t = c_seq[:, :-k]       # [B, T-k, H]
            pred = nn.Dense(self.projection_dim, name=f"predictor_{k}")(c_t)

            logits = jnp.einsum("btd,bkd->btk", pred, z_pos) / self.temperature
            labels = jnp.arange(logits.shape[1])
            loss += jnp.mean(
                -jax.nn.log_softmax(logits, axis=-1)[..., labels]
            )

        return loss / self.num_future_steps