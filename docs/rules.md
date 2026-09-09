# Rules

Both dimensions use toroidal Moore neighborhoods. 2D rules use counts from
`0..8`; 3D rules use all `26` surrounding voxels and counts from `0..26`.

Rules are written as `B`irth and `S`urvival counts. The 2D Conway preset is
`B3/S23`. In 3D, multiple counts must be comma-separated, for example
`B6/S5,6,7`; `S23` means exactly 23 neighbors.

The parser and formatter live in `emergent.core.rules` and
`emergent.core3d.rules`. The numerical functions receive parsed rule values or
explicit JAX masks rather than parsing inside compiled code.
