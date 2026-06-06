# Legacy scene_info split path

This directory preserves the Day 2 reverse-splitting implementation and its
generated episode lists.

The production path is now `xiaoma26/calvin-lerobot`, which provides official
`splitA`, `splitB`, `splitC`, and `splitD` directories. The old
`scene_info.npy` path should not be used to create training inputs anymore.

The old audit remains useful as cross-validation evidence: the independent
frame-range audit reached the same official training split counts for
`A=6089`, `B=6115`, and `C=5666`, with `cross=0` and `orphan=0`.
