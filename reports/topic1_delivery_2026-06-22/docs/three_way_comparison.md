# Three-Way Comparison For Report

| Route | Asset | Geometry Quality | Texture Quality | Time / Cost | Failure Modes / Limitations | Final Form |
|---|---|---|---|---|---|---|
| Real multi-view reconstruction | object A | Best physical consistency because it is reconstructed from real captured views. 3DGS 30k reaches test PSNR 31.3322. | Good under original camera views; mesh texturing requires clean foreground and good view coverage. | Requires phone capture, frame extraction, COLMAP, 3DGS, dense meshing, and texrecon. More manual work than B/C. | Early point cloud/mesh contained background clutter; needed manual cleanup (`clean_A.ply`) before texture reconstruction. | 3DGS `.ply` and textured mesh (`object_A_textured.obj/.mtl`). |
| Text-to-3D | object B hamburger | Recognizable global hamburger shape after 10000 iterations, but local geometry may be soft or imperfect. | Has exported texture and is Blender-readable. Texture is plausible but generated rather than photo-consistent. | No capture data needed; one 100-step smoke run plus one 10000-step full run. | Prompt/seed sensitivity, SDS blur, possible local artifacts. | Textured mesh (`model.obj`, `model.mtl`, `texture_kd.jpg`). |
| Single-image-to-3D | object C toy | Captures the main silhouette from one RGBA image. 3000 iter result was selected as the best practical version. | Texture is usable and exported with the mesh; quality depends heavily on input crop/matte. | Lowest data cost; requires image crop, alpha matte, Stable Zero123 training, and export. | Sensitive to input segmentation; 10000 iter did not clearly improve visual quality and no mesh export was found. | 3000 iter textured mesh (`model.obj`, `model.mtl`, `texture_kd.jpg`). |

Short conclusion for the report:

Object A is the strongest route for geometry fidelity when capture quality is good. Object B offers the highest semantic freedom because it starts from text. Object C has the lowest capture cost and is useful when only a single photo is available, but its quality depends strongly on the input crop and foreground alpha.

