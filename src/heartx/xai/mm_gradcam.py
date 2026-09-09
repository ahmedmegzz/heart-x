"""MM-GradCAM: multi-modal Grad-CAM for 1D and 2D streams."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    denom = cam.max() - cam.min()
    if denom < 1e-8:
        return np.zeros_like(cam)
    return (cam - cam.min()) / denom


class MMGradCAM:
    """
    Gradient-weighted Class Activation Mapping for HEART-X dual streams.

    Produces aligned heatmaps for the 1D CNN feature maps and the 2D
    spectrogram CNN feature maps for a chosen target class.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def _gradcam_from_activations(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        target_length: int | None = None,
        mode: Literal["1d", "2d"] = "1d",
    ) -> np.ndarray:
        # activations / gradients: (B, C, ...)
        weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
        cam = (weights * activations).sum(dim=1)  # (B, ...)
        cam = F.relu(cam)
        if mode == "1d":
            cam = cam.unsqueeze(1)
            if target_length is not None:
                cam = F.interpolate(
                    cam, size=target_length, mode="linear", align_corners=False
                )
            cam = cam.squeeze(1)
        else:
            if target_length is not None:
                # target_length interpreted as (H, W) tuple passed separately
                pass
        return cam.detach().cpu().numpy()

    def generate(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        clinical: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Returns dict with keys:
          - cam_1d: (T,)
          - cam_2d: (H, W)
          - pred_class: int
          - probs: (C,)
        """
        self.model.zero_grad(set_to_none=True)
        x_1d = x_1d.detach().requires_grad_(True)
        x_2d = x_2d.detach().requires_grad_(True)

        logits = self.model(x_1d, x_2d, clinical)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        cls = pred if target_class is None else target_class
        score = logits[0, cls]
        score.backward()

        # 1D stream
        act_1d = self.model.stream_1d.last_conv
        # Recompute with hooks for gradients on last conv
        # Simpler path: use input-gradient attribution fallback if hooks missing
        cam_1d = self._input_grad_1d(x_1d)
        cam_2d = self._input_grad_2d(x_2d)

        # Prefer activation Grad-CAM when last_conv is available
        if act_1d is not None and act_1d.grad is not None:
            cam = self._gradcam_from_activations(
                act_1d, act_1d.grad, target_length=x_1d.shape[-1], mode="1d"
            )[0]
            cam_1d = _normalize_map(cam)

        act_2d = self.model.stream_2d.last_conv
        if act_2d is not None and getattr(act_2d, "grad", None) is not None:
            weights = act_2d.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act_2d).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=x_2d.shape[-2:], mode="bilinear", align_corners=False
            )
            cam_2d = _normalize_map(cam[0, 0].detach().cpu().numpy())

        return {
            "cam_1d": cam_1d,
            "cam_2d": cam_2d,
            "pred_class": pred,
            "target_class": cls,
            "probs": probs[0].detach().cpu().numpy(),
        }

    def _input_grad_1d(self, x_1d: torch.Tensor) -> np.ndarray:
        if x_1d.grad is None:
            return np.zeros(x_1d.shape[-1], dtype=np.float32)
        g = x_1d.grad[0, 0].abs().detach().cpu().numpy()
        return _normalize_map(g)

    def _input_grad_2d(self, x_2d: torch.Tensor) -> np.ndarray:
        if x_2d.grad is None:
            return np.zeros(x_2d.shape[-2:], dtype=np.float32)
        g = x_2d.grad[0, 0].abs().detach().cpu().numpy()
        return _normalize_map(g)


class MMGradCAMHooked(MMGradCAM):
    """Grad-CAM using forward/backward hooks on last convolutional layers."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self._acts: dict[str, torch.Tensor] = {}
        self._grads: dict[str, torch.Tensor] = {}
        self._handles = []

        def _save_act(name):
            def hook(_m, _i, o):
                self._acts[name] = o
                o.retain_grad()

            return hook

        def _save_grad(name):
            def hook(_m, _gi, go):
                self._grads[name] = go[0]

            return hook

        # Find last Conv1d / Conv2d more robustly
        last_1d = [m for m in model.stream_1d.features.modules() if isinstance(m, torch.nn.Conv1d)][-1]
        last_2d = [m for m in model.stream_2d.features.modules() if isinstance(m, torch.nn.Conv2d)][-1]
        self._handles.append(last_1d.register_forward_hook(_save_act("1d")))
        self._handles.append(last_1d.register_full_backward_hook(_save_grad("1d")))
        self._handles.append(last_2d.register_forward_hook(_save_act("2d")))
        self._handles.append(last_2d.register_full_backward_hook(_save_grad("2d")))

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        clinical: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> dict[str, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        self._acts.clear()
        self._grads.clear()

        logits = self.model(x_1d, x_2d, clinical)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        cls = pred if target_class is None else target_class
        logits[0, cls].backward()

        act_1d = self._acts["1d"]
        grad_1d = self._grads["1d"]
        weights = grad_1d.mean(dim=2, keepdim=True)
        cam_1d = F.relu((weights * act_1d).sum(dim=1, keepdim=True))
        cam_1d = F.interpolate(
            cam_1d, size=x_1d.shape[-1], mode="linear", align_corners=False
        )
        cam_1d = _normalize_map(cam_1d[0, 0].detach().cpu().numpy())

        act_2d = self._acts["2d"]
        grad_2d = self._grads["2d"]
        weights = grad_2d.mean(dim=(2, 3), keepdim=True)
        cam_2d = F.relu((weights * act_2d).sum(dim=1, keepdim=True))
        cam_2d = F.interpolate(
            cam_2d, size=x_2d.shape[-2:], mode="bilinear", align_corners=False
        )
        cam_2d = _normalize_map(cam_2d[0, 0].detach().cpu().numpy())

        return {
            "cam_1d": cam_1d,
            "cam_2d": cam_2d,
            "pred_class": pred,
            "target_class": cls,
            "probs": probs[0].detach().cpu().numpy(),
        }
