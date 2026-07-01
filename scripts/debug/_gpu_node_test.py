"""Quick GPU-node smoke test: validates the CUDA driver + PYRO-NN projector +
512^2 backprop at the Mayo geometry on whatever node Slurm placed us on.
Run via: srun --nodelist=lmeXX --gres=gpu:1 ... python scripts/debug/_gpu_node_test.py
No dataset load (geometry only) so it finishes in ~1 min. Prints OK on success."""
import os, socket, sys
sys.path.insert(0, "/cluster/maier/Agent4CT")
import torch

print(f"[node] host={socket.gethostname()}", flush=True)
assert torch.cuda.is_available(), "CUDA not available — driver/torch mismatch"
print(f"[gpu]  {torch.cuda.get_device_name(0)}  "
      f"capability={torch.cuda.get_device_capability(0)}", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"[vram] total={total/1e9:.1f}GB free={free/1e9:.1f}GB", flush=True)

# Representative Mayo recon workload: forward/back projection at full geometry + backprop.
from ddssl_ldct.staged_dataset import GEOMETRIES
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.models import SmallUNet

info = GEOMETRIES["mayo_ldct_2d"]
geom = FanBeamGeometry(image_size=512, pixel_spacing=info.pixel_spacing,
                       n_angles=info.n_angles, n_det=info.n_det,
                       det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd)
proj = PyronnFanBeamProjector(geom).cuda()
x = torch.rand(2, 1, 512, 512, device="cuda", requires_grad=True)
sino = proj.forward_project(x)
bp = proj.back_project(sino)
print(f"[proj] fwd {tuple(sino.shape)} -> bp {tuple(bp.shape)}", flush=True)

net = SmallUNet(c=16).cuda()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for _ in range(3):
    opt.zero_grad()
    fbp = torch.clamp(proj.fbp(sino), min=0.0)
    loss = (net(fbp) - x).pow(2).mean() + bp.mean() * 0.0
    loss.backward()
    opt.step()
torch.cuda.synchronize()
print(f"[train] 3 steps OK, last loss={float(loss):.4e}", flush=True)
print(f"[vram] peak_allocated={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
print("OK", flush=True)
