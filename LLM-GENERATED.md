# About this branch

This is a personal fork. The NAK/NVK commits on this branch (and any commit
with a `Generated-by: LLM` trailer) were written by an AI model (Claude),
directed by the fork's owner. They are kept here for reference and
experimentation only.

They are not intended for upstream Mesa. Mesa's AGENTS.md and the nouveau
project's policy (https://nouveau.freedesktop.org/Humanity.html) do not accept
LLM-generated contributions to `src/nouveau`. If you use any of this as
inspiration for upstream work, write your own implementation.

What is here, on top of upstream main:
- Dave Airlie's open MR !41775 (workgroup scope cooperative matrices, decode
  vectors), cherry-picked, plus one fix to its split get_coordinate.
- VK_NV_cooperative_matrix2 for NVK (Turing and newer): NAK lowering of
  tensor addressing, block loads, reductions, per-element operations,
  conversions, transposes and coordinates on top of Mesa's shared
  flexible-dimension and workgroup-scope lowering.
