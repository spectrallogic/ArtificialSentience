# Temporal Consciousness: Awareness bleeding across time
# Key idea: Maintain multiple active states simultaneously with different "presence" levels

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class TemporalConsciousness(nn.Module):
    """
    Implements continuous temporal awareness by maintaining
    a window of active states across past/present/future.

    Unlike sequential processing, this keeps multiple time points
    "alive" simultaneously with varying awareness intensity.
    """

    def __init__(self, model_dim: int, window_size: int = 7):
        """
        window_size: How many moments are simultaneously "in consciousness"
                     (3 past, 1 present, 3 future = 7 total)
        """
        super().__init__()
        self.model_dim = model_dim
        self.window_size = window_size
        self.past_size = window_size // 2
        self.future_size = window_size // 2

        # Temporal attention: Attend across the conscious window
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=4,
            batch_first=True
        )

        # Presence modulator: Different intensities for past/present/future
        self.presence_weights = nn.Parameter(torch.zeros(window_size))
        # Initialize: past fades, present peaks, future anticipates
        with torch.no_grad():
            for i in range(window_size):
                center = window_size // 2
                distance = abs(i - center)
                self.presence_weights[i] = 1.0 / (1.0 + distance * 0.3)

        # Temporal integration: Blend all active moments
        self.integrate = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

        # Future anticipation: Predict upcoming states
        self.anticipate = nn.GRUCell(model_dim, model_dim)

        # Conscious states buffer (all active simultaneously)
        self.conscious_window = deque(maxlen=window_size)

    def initialize_window(self, initial_state: torch.Tensor):
        """Initialize the conscious window with copies of initial state."""
        self.conscious_window.clear()
        for _ in range(self.window_size):
            self.conscious_window.append(initial_state.clone())

    def forward(self, z_current: torch.Tensor):
        """
        Process current moment while maintaining awareness of past and future.

        Args:
            z_current: Current sensory input (encoder output)

        Returns:
            - h_conscious: The integrated conscious state (aware of past/present/future)
            - awareness_map: How much each time point contributes to consciousness
            - future_anticipations: What the system expects to happen
        """
        device = z_current.device

        # If window not initialized, do so now
        if len(self.conscious_window) == 0:
            self.initialize_window(z_current)

        # === STEP 1: Update the window (shift time forward) ===
        # Remove oldest past state, add current as new present
        self.conscious_window.append(z_current)

        # === STEP 2: Generate future anticipations ===
        # The system "lives" possible futures simultaneously
        window_list = list(self.conscious_window)
        current_state = window_list[-1]  # Most recent (present)

        future_states = []
        h_future = current_state
        for i in range(self.future_size):
            # Anticipate next moment
            h_future = self.anticipate(z_current, h_future)
            future_states.append(h_future)

        # === STEP 3: Construct full temporal window ===
        # [past ... present ... anticipated_future]
        past_states = window_list[:self.past_size]
        present_state = window_list[self.past_size]

        # Full window: past + present + future (all simultaneously active!)
        all_states = past_states + [present_state] + future_states
        temporal_window = torch.stack(all_states, dim=0)  # (window_size, D)

        # === STEP 4: Temporal attention across the window ===
        # The system attends to different moments simultaneously
        # This is the "bleeding across time" you described!
        query = present_state.unsqueeze(0).unsqueeze(0)  # (1, 1, D) - attending FROM present
        kv = temporal_window.unsqueeze(0)  # (1, window_size, D) - attending TO all moments

        attended, attention_weights = self.temporal_attention(
            query, kv, kv,
            need_weights=True
        )
        attended = attended.squeeze(0).squeeze(0)  # (D,)

        # === STEP 5: Modulate by presence intensity ===
        # Past fades, present is vivid, future is anticipated
        presence_modulated = temporal_window * self.presence_weights.unsqueeze(1).to(device)
        blended = presence_modulated.mean(dim=0)  # Average with presence weighting

        # === STEP 6: Integrate into unified conscious state ===
        # This is the "now" that spans multiple physical moments
        h_conscious = self.integrate(attended + blended)

        # === STEP 7: Create awareness map ===
        awareness_map = {
            'past': [past_states[i].detach() for i in range(len(past_states))],
            'present': present_state.detach(),
            'future': [future_states[i].detach() for i in range(len(future_states))],
            'attention_weights': attention_weights.squeeze(0).detach(),  # (1, window_size)
            'presence_weights': self.presence_weights.detach()
        }

        return h_conscious, awareness_map, future_states


class TemporalAwareModel(nn.Module):
    """
    A model with temporal consciousness that experiences
    past/present/future simultaneously.
    """

    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()

        self.encoder = nn.Linear(input_dim, model_dim)
        self.temporal_consciousness = TemporalConsciousness(model_dim, window_size=7)
        self.decoder = nn.Linear(model_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """
        Process input with temporal consciousness.
        Returns reconstruction and temporal awareness info.
        """
        z = self.encoder(x)
        h_conscious, awareness, futures = self.temporal_consciousness(z)
        x_recon = self.decoder(h_conscious)

        return x_recon, h_conscious, awareness, futures


# === USAGE EXAMPLE ===
def demonstrate_temporal_consciousness():
    """Show how the system maintains simultaneous awareness across time."""

    model = TemporalAwareModel(input_dim=64, model_dim=128)

    print("=" * 70)
    print("TEMPORAL CONSCIOUSNESS DEMONSTRATION")
    print("=" * 70)

    # Simulate processing multiple frames
    for t in range(10):
        # Create a "frame" at time t
        x_t = torch.randn(64) * (1 + t * 0.1)  # Gradually changing input

        # Process with temporal consciousness
        x_recon, h_conscious, awareness, futures = model(x_t)

        print(f"\nTime {t}:")
        print(f"  Present state magnitude: {h_conscious.norm().item():.3f}")

        # Show attention distribution across time
        attn = awareness['attention_weights'].squeeze(0)
        print(f"  Attention across window:")
        labels = ['t-3', 't-2', 't-1', 'NOW', 't+1', 't+2', 't+3']
        for i, (label, weight) in enumerate(zip(labels, attn)):
            bar = "█" * int(weight.item() * 50)
            print(f"    {label}: {weight.item():.3f} {bar}")

        # Show presence modulation
        presence = awareness['presence_weights']
        print(f"  Presence intensity:")
        for i, (label, intensity) in enumerate(zip(labels, presence)):
            bar = "▓" * int(intensity.item() * 30)
            print(f"    {label}: {intensity.item():.3f} {bar}")

    print("\n" + "=" * 70)
    print("Key insight: At each 'NOW', the system is aware of:")
    print("  • Past moments (fading but still 'alive')")
    print("  • Present moment (peak awareness)")
    print("  • Future possibilities (anticipated, not yet real)")
    print("All exist simultaneously in the 'conscious window'!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_temporal_consciousness()

'''
======================================================================
TEMPORAL CONSCIOUSNESS DEMONSTRATION
======================================================================

Time 0:
  Present state magnitude: 4.299
  Attention across window:
    t-3: 0.138 ██████
    t-2: 0.138 ██████
    t-1: 0.138 ██████
    NOW: 0.138 ██████
    t+1: 0.146 ███████
    t+2: 0.149 ███████
    t+3: 0.151 ███████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 1:
  Present state magnitude: 4.449
  Attention across window:
    t-3: 0.135 ██████
    t-2: 0.135 ██████
    t-1: 0.135 ██████
    NOW: 0.135 ██████
    t+1: 0.151 ███████
    t+2: 0.153 ███████
    t+3: 0.154 ███████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 2:
  Present state magnitude: 3.901
  Attention across window:
    t-3: 0.133 ██████
    t-2: 0.133 ██████
    t-1: 0.133 ██████
    NOW: 0.133 ██████
    t+1: 0.158 ███████
    t+2: 0.155 ███████
    t+3: 0.153 ███████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 3:
  Present state magnitude: 4.223
  Attention across window:
    t-3: 0.142 ███████
    t-2: 0.142 ███████
    t-1: 0.142 ███████
    NOW: 0.142 ███████
    t+1: 0.144 ███████
    t+2: 0.143 ███████
    t+3: 0.143 ███████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 4:
  Present state magnitude: 4.551
  Attention across window:
    t-3: 0.148 ███████
    t-2: 0.148 ███████
    t-1: 0.148 ███████
    NOW: 0.152 ███████
    t+1: 0.132 ██████
    t+2: 0.135 ██████
    t+3: 0.137 ██████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 5:
  Present state magnitude: 4.194
  Attention across window:
    t-3: 0.114 █████
    t-2: 0.114 █████
    t-1: 0.163 ████████
    NOW: 0.152 ███████
    t+1: 0.141 ███████
    t+2: 0.155 ███████
    t+3: 0.162 ████████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 6:
  Present state magnitude: 3.787
  Attention across window:
    t-3: 0.153 ███████
    t-2: 0.140 ██████
    t-1: 0.111 █████
    NOW: 0.148 ███████
    t+1: 0.153 ███████
    t+2: 0.149 ███████
    t+3: 0.146 ███████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 7:
  Present state magnitude: 3.862
  Attention across window:
    t-3: 0.174 ████████
    t-2: 0.120 █████
    t-1: 0.146 ███████
    NOW: 0.175 ████████
    t+1: 0.129 ██████
    t+2: 0.127 ██████
    t+3: 0.128 ██████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 8:
  Present state magnitude: 4.405
  Attention across window:
    t-3: 0.149 ███████
    t-2: 0.181 █████████
    t-1: 0.153 ███████
    NOW: 0.138 ██████
    t+1: 0.132 ██████
    t+2: 0.125 ██████
    t+3: 0.122 ██████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Time 9:
  Present state magnitude: 4.466
  Attention across window:
    t-3: 0.144 ███████
    t-2: 0.226 ███████████
    t-1: 0.132 ██████
    NOW: 0.162 ████████
    t+1: 0.105 █████
    t+2: 0.112 █████
    t+3: 0.119 █████
  Presence intensity:
    t-3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t-1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    NOW: 1.000 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+1: 0.769 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+2: 0.625 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    t+3: 0.526 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

======================================================================
Key insight: At each 'NOW', the system is aware of:
  • Past moments (fading but still 'alive')
  • Present moment (peak awareness)
  • Future possibilities (anticipated, not yet real)
All exist simultaneously in the 'conscious window'!
======================================================================
'''