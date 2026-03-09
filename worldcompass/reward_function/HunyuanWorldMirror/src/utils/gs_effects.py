from math import atan2, cos, exp, floor, sin, sqrt

import numpy as np
import torch


def fract(x):
    """Get fractional part of a number."""
    if isinstance(x, torch.Tensor):
        return x - torch.floor(x)
    return x - floor(x)


class GSEffects:
    """Convert GLSL GS render effects to PyTorch - vectorized for batch processing"""

    def __init__(self, start_time=0.0, end_time=10.0):
        """Initialize effects with time range.

        Args:
            start_time: Animation start time
            end_time: Animation end time
        """
        self.start_time = start_time
        self.end_time = end_time

    @staticmethod
    def smoothstep(edge0, edge1, x):
        """GLSL smoothstep function (vectorized)"""
        if isinstance(x, torch.Tensor):
            result = torch.zeros_like(x, dtype=x.dtype)
            mask_low = x < edge0
            mask_high = x > edge1
            mask_mid = ~(mask_low | mask_high)

            t = (x[mask_mid] - edge0) / (edge1 - edge0)
            result[mask_mid] = t * t * (3.0 - 2.0 * t)
            result[mask_low] = 0.0
            result[mask_high] = 1.0
            return result
        else:
            if x < edge0:
                return 0.0
            if x > edge1:
                return 1.0
            t = (x - edge0) / (edge1 - edge0)
            return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def step(edge, x):
        """GLSL step function (vectorized)"""
        if isinstance(x, torch.Tensor):
            return (x >= edge).to(x.dtype)
        if isinstance(edge, torch.Tensor):
            return (x >= edge).to(edge.dtype)
        return 1.0 if x >= edge else 0.0

    @staticmethod
    def mix(x, y, a):
        """GLSL mix function (linear interpolation, vectorized)"""
        return x * (1.0 - a) + y * a

    @staticmethod
    def clamp(x, min_val, max_val):
        """Clamp value between min and max (vectorized)"""
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        return max(min_val, min(max_val, x))

    @staticmethod
    def length_xz(pos):
        """Calculate length of XZ components (vectorized)"""
        if pos.dim() == 1:
            return torch.sqrt(pos[0] ** 2 + pos[2] ** 2)
        return torch.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    @staticmethod
    def length_vec(v):
        """Calculate vector length (vectorized)"""
        if v.dim() == 1:
            return torch.sqrt(torch.sum(v**2))
        return torch.sqrt(torch.sum(v**2, dim=1))

    @staticmethod
    def hash(p):
        """Pseudo-random hash function (vectorized)"""
        p = fract(p * 0.3183099 + 0.1)
        p = p * 17.0
        return torch.stack(
            [
                fract(p[:, 0] * p[:, 1] * p[:, 2]),
                fract(p[:, 0] + p[:, 1] * p[:, 2]),
                fract(p[:, 0] * p[:, 1] + p[:, 2]),
            ],
            dim=1,
        )

    @staticmethod
    def noise(p):
        """3D Perlin-style noise function (vectorized)"""
        i = torch.floor(p).to(torch.long)
        f = fract(p)
        f = f * f * (3.0 - 2.0 * f)

        def get_hash_offset(offset):
            return GSEffects.hash(i.to(p.dtype) + offset)

        n000 = get_hash_offset(
            torch.tensor([0, 0, 0], dtype=p.dtype, device=p.device)
        )
        n100 = get_hash_offset(
            torch.tensor([1, 0, 0], dtype=p.dtype, device=p.device)
        )
        n010 = get_hash_offset(
            torch.tensor([0, 1, 0], dtype=p.dtype, device=p.device)
        )
        n110 = get_hash_offset(
            torch.tensor([1, 1, 0], dtype=p.dtype, device=p.device)
        )
        n001 = get_hash_offset(
            torch.tensor([0, 0, 1], dtype=p.dtype, device=p.device)
        )
        n101 = get_hash_offset(
            torch.tensor([1, 0, 1], dtype=p.dtype, device=p.device)
        )
        n011 = get_hash_offset(
            torch.tensor([0, 1, 1], dtype=p.dtype, device=p.device)
        )
        n111 = get_hash_offset(
            torch.tensor([1, 1, 1], dtype=p.dtype, device=p.device)
        )

        x0 = GSEffects.mix(n000, n100, f[:, 0:1])
        x1 = GSEffects.mix(n010, n110, f[:, 0:1])
        x2 = GSEffects.mix(n001, n101, f[:, 0:1])
        x3 = GSEffects.mix(n011, n111, f[:, 0:1])

        y0 = GSEffects.mix(x0, x1, f[:, 1:2])
        y1 = GSEffects.mix(x2, x3, f[:, 1:2])

        return GSEffects.mix(y0, y1, f[:, 2:3])

    @staticmethod
    def rot_2d(angle):
        """2D rotation (vectorized)"""
        if isinstance(angle, torch.Tensor):
            s = torch.sin(angle)
            c = torch.cos(angle)
            rot = torch.stack(
                [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
                dim=-2,
            ).squeeze()
        else:
            s = np.sin(angle)
            c = np.cos(angle)
            rot = torch.tensor([[c, -s], [s, c]]).cuda().float()
        return rot

    def twister(self, pos, scale, t):
        h = self.hash(pos)[:, 0:1] + 0.1
        pos_xz_len = self.length_xz(pos)
        s = self.smoothstep(0.0, 8.0, t * t * 0.1 - pos_xz_len * 2.0 + 2.0)[
            :, None
        ]
        mask = torch.linalg.norm(scale, dim=-1, keepdim=True) < 0.05
        pos_y = torch.where(
            mask, (-10.0 + pos[:, 1:2]) * (s ** (2 * h)), pos[:, 1:2]
        )
        pos_xz = pos[:, [0, 2]] * torch.exp(
            -1 * torch.linalg.norm(pos[:, [0, 2]], dim=-1, keepdim=True)
        )
        pos_xz = torch.einsum(
            "n i, n i j -> n j",
            pos_xz,
            self.rot_2d(t * 0.2 + pos[:, 1:2] * 20.0 * (1 - s)),
        )
        pos_new = torch.cat([pos_xz[:, 0:1], pos_y, pos_xz[:, 1:2]], dim=-1)
        return pos_new, s**4

    def rain(self, pos, scale, t):
        h = self.hash(pos)
        pos_xz_len = self.length_xz(pos)
        s = self.smoothstep(0.0, 5.0, t * t * 0.1 - pos_xz_len * 2.0 + 1.0) ** (
            0.5 + h[:, 0]
        )
        y = pos[:, 1:2]
        pos_y = torch.minimum(-10.0 + s[:, None] * 15.0, pos[:, 1:2])
        pos_x = pos[:, 0:1] + pos_y * 0.2
        pos_xz = torch.cat([pos_x, pos[:, 2:3]], dim=-1)
        pos_xz = pos_xz * torch.matmul(
            self.rot_2d(t * 0.3), torch.ones_like(pos_xz).unsqueeze(-1)
        ).squeeze(-1)
        pos_new = torch.cat([pos_xz[:, 0:1], pos_y, pos_xz[:, 1:2]], dim=-1)
        a = self.smoothstep(-10.0, y.squeeze(), pos_y.squeeze())[:, None]
        return pos_new, a

    def apply_effect(self, gsplat, t, effect_type, ignore_scale=False):
        """Apply the effect shader logic (vectorized for batch processing)

        Args:
            gsplat: Dictionary with:
                'means': (n, 3) tensor
                'scales': (n, 3) tensor
                'colors': (n, 3) tensor
                'quats': (n, 4) tensor
                'opacities': (n,) tensor
            t: Current time (normalized based on start_time and end_time)
            effect_type: 2=Spread

        Returns:
            Modified gsplat dictionary
        """
        # Normalize time to animation range
        normalized_t = t - self.start_time
        device = gsplat["means"].device
        dtype = gsplat["means"].dtype

        output = {
            "means": gsplat["means"].clone(),
            "quats": gsplat["quats"].clone(),
            "scales": gsplat["scales"].clone(),
            "opacities": gsplat["opacities"].clone(),
            "colors": gsplat["colors"].clone(),
        }

        s = self.smoothstep(0.0, 10.0, normalized_t - 3.2) * 10.0
        scales = output["scales"]
        local_pos = output["means"].clone()
        l = self.length_xz(local_pos)
        smoothstep_val = None

        if effect_type == 2:  # Spread Effect
            border = torch.abs(s - l - 0.5)
            decay = 1.0 - 0.2 * torch.exp(-20.0 * border)
            # decay = 1.0 - 0.7 * torch.exp(-10.0 * border)
            local_pos = local_pos * decay[:, None]

            smoothstep_val = self.smoothstep(s - 0.5, s, l + 0.5)
            # final_scales = self.mix(scales, 0.002, smoothstep_val[:, None])
            if not ignore_scale:
                final_scales = self.mix(scales, 1e-9, smoothstep_val[:, None])
            else:
                final_scales = scales

            noise_input = torch.stack(
                [
                    local_pos[:, 0] * 2.0 + normalized_t * 0.5,
                    local_pos[:, 1] * 2.0 + normalized_t * 0.5,
                    local_pos[:, 2] * 2.0 + normalized_t * 0.5,
                ],
                dim=1,
            )
            noise_val = self.noise(noise_input)

            output["means"] = (
                local_pos + 0.0 * noise_val * smoothstep_val[:, None]
            )
            output["scales"] = final_scales

            at = torch.atan2(local_pos[:, 0], local_pos[:, 2]) / 3.1416
            output["colors"] *= self.step(at, normalized_t - 3.1416)[:, None]
            output["colors"] += (
                torch.exp(-20.0 * border)
                + torch.exp(-50.0 * torch.abs(normalized_t - at - 3.1416)) * 0.5
            )[:, None]
            output["opacities"] *= self.step(at, normalized_t - 3.1416)
            output["opacities"] += (
                torch.exp(-20.0 * border)
                + torch.exp(-50.0 * torch.abs(normalized_t - at - 3.1416)) * 0.5
            )

            # ===== New feature: Randomly mask points based on smoothstep_val =====
            # Higher smoothstep_val means higher probability of masking
            mask_prob = (
                smoothstep_val.squeeze()
                if smoothstep_val.dim() > 1
                else smoothstep_val
            )
            if not hasattr(self, "random_vals"):
                self.random_vals = torch.rand(
                    mask_prob.shape, device=device, dtype=dtype
                )
            mask = (
                self.random_vals < mask_prob * 0.8
            )  # True indicates the point is masked

            # Apply mask to various attributes
            if not ignore_scale:
                output["means"][mask] *= 0  # Or can be set to other values
                output["scales"][
                    mask
                ] *= 0  # Set scales to 0 to make points invisible
                output["opacities"][
                    mask
                ] *= 0  # Set opacity to 0 to make points transparent

        return output, smoothstep_val


# Usage example
if __name__ == "__main__":
    # Create effects processor with time range from 0 to 10 seconds
    effects = GSEffects(start_time=0.0, end_time=10.0)

    # Sample gsplat data (batch)
    n_points = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_gsplat = {
        "means": torch.randn(n_points, 3, dtype=torch.float32, device=device),
        "quats": torch.randn(n_points, 4, dtype=torch.float32, device=device),
        "scales": torch.rand(n_points, 3, dtype=torch.float32, device=device),
        "opacities": torch.rand(n_points, dtype=torch.float32, device=device),
        "colors": torch.rand(n_points, 3, dtype=torch.float32, device=device),
    }

    # Apply Magic effect at different time points
    for t in [0.0, 2.5, 5.0, 7.5, 10.0]:
        result = effects.apply_effect(sample_gsplat, t, effect_type=2)
        print(f"\nTime: {t}s")
        print(f"Center shape: {result['means'].shape}")
        print(f"Center[0]: {result['means'][0]}")
        print(f"Scales shape: {result['scales'].shape}")
        print(f"Scales[0]: {result['scales'][0]}")
        print(f"RGB shape: {result['colors'].shape}")
        print(f"RGB[0]: {result['colors'][0]}")
        print(f"Opacity shape: {result['opacities'].shape}")
        print(f"Opacity[0]: {result['opacities'][0]}")
