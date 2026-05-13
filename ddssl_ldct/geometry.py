"""2D fan-beam geometry matching Wagner et al. 2023 (arXiv:2211.01111).

Defaults reproduce the rebinned helical Mayo LDCT-and-Projection-data geometry
(Siemens SOMATOM Definition AS in the DICOM-CT-PD format), as used by the
`helix2fan` repository's reference reconstruction (`reco_example_fan_beam.py`):

  - image: 512 × 512 at 0.7 mm voxel pitch
  - detector: 736 channels, 1.2858 mm spacing (flat detector after rebinning)
  - source-isocentre distance (SOD / dso): 595.0 mm
  - source-detector distance (SDD / dsd):   1085.6 mm
  - 1152 views per full rotation (one-rotation full scan, 2π)

For the AAPM 2016 Low Dose CT Grand Challenge the slices are reconstructed at
1 mm thickness with these in-plane parameters (cf. Wagner 2022 Med. Phys.).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class FanBeamGeometry:
    image_size: int = 512
    pixel_spacing: float = 0.7              # mm per voxel (in-slice)
    n_angles: int = 1152                    # one full rotation
    n_det: int = 736                        # Siemens AS channels after rebinning
    det_spacing: float = 1.2858             # mm per detector pixel
    sod: float = 595.0                      # source-object distance, mm
    sdd: float = 1085.6                     # source-detector distance, mm
    angle_start: float = 0.0
    angle_end: float = 2 * math.pi          # full-scan

    @property
    def odd(self) -> float:
        return self.sdd - self.sod

    def angles(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.linspace(self.angle_start, self.angle_end, self.n_angles + 1,
                              device=device, dtype=dtype)[:-1]

    def det_u(self, device=None, dtype=torch.float32) -> torch.Tensor:
        n = self.n_det
        u = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0) * self.det_spacing
        return u

    def split_angles(self) -> tuple["FanBeamGeometry", "FanBeamGeometry"]:
        """Even / odd angular subsets, used for Noise2Inverse-style splits."""
        even = FanBeamGeometry(**{**self.__dict__, "n_angles": self.n_angles // 2})
        odd = FanBeamGeometry(**{**self.__dict__, "n_angles": self.n_angles // 2})
        return even, odd
