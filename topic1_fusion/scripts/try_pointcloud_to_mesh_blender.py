import bpy
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
ply_path = repo_root / "topic1_fusion/outputs/object_A/gs_30k/point_cloud/iteration_30000/point_cloud.ply"
out_blend = repo_root / "topic1_fusion/outputs/object_A/gs_30k/object_A_mesh_attempt.blend"
out_obj = repo_root / "topic1_fusion/outputs/object_A/gs_30k/object_A_mesh_attempt.obj"

# clean default scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# enable obj export if needed
try:
    bpy.ops.preferences.addon_enable(module="io_scene_obj")
except Exception:
    pass

# import point cloud ply
bpy.ops.import_mesh.ply(filepath=str(ply_path))
obj = bpy.context.selected_objects[0]
obj.name = "ObjectA_PointCloud"

# geometry nodes: Mesh to Points -> Points to Volume -> Volume to Mesh
mod = obj.modifiers.new(name="GN_MeshApprox", type='NODES')
ng = bpy.data.node_groups.new("PointCloudToMesh", 'GeometryNodeTree')
mod.node_group = ng

# interface
try:
    ng.inputs.new('NodeSocketGeometry', 'Geometry')
    ng.outputs.new('NodeSocketGeometry', 'Geometry')
except Exception:
    pass

nodes = ng.nodes
links = ng.links
for n in list(nodes):
    nodes.remove(n)

in_node = nodes.new('NodeGroupInput')
out_node = nodes.new('NodeGroupOutput')
mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
points_to_volume = nodes.new('GeometryNodePointsToVolume')
volume_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
set_material = nodes.new('GeometryNodeSetMaterial')

# layout
in_node.location = (-900, 0)
mesh_to_points.location = (-650, 0)
points_to_volume.location = (-350, 0)
volume_to_mesh.location = (-50, 0)
set_material.location = (200, 0)
out_node.location = (450, 0)

# parameters
try:
    mesh_to_points.inputs['Radius'].default_value = 0.003
except Exception:
    pass
try:
    points_to_volume.inputs['Density'].default_value = 1.0
except Exception:
    pass
try:
    points_to_volume.inputs['Voxel Size'].default_value = 0.004
except Exception:
    pass
try:
    points_to_volume.inputs['Radius'].default_value = 0.01
except Exception:
    pass
try:
    volume_to_mesh.inputs['Voxel Size'].default_value = 0.004
except Exception:
    pass
try:
    volume_to_mesh.inputs['Threshold'].default_value = 0.1
except Exception:
    pass

mat = bpy.data.materials.new(name='ObjectA_MeshMat')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.8, 0.45, 0.6, 1.0)

set_material.inputs['Material'].default_value = mat

links.new(in_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
links.new(mesh_to_points.outputs['Points'], points_to_volume.inputs['Points'])
links.new(points_to_volume.outputs['Volume'], volume_to_mesh.inputs['Volume'])
links.new(volume_to_mesh.outputs['Mesh'], set_material.inputs['Geometry'])
links.new(set_material.outputs['Geometry'], out_node.inputs['Geometry'])

# evaluate and realize modifier by converting to mesh
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.modifier_apply(modifier=mod.name)
obj.name = 'ObjectA_MeshAttempt'

# save/export
bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
try:
    bpy.ops.export_scene.obj(filepath=str(out_obj), use_selection=True, use_materials=True)
except Exception as e:
    print('OBJ export failed', e)

mesh = obj.data
print('RESULT', obj.name, len(mesh.vertices), len(mesh.edges), len(mesh.polygons))
print('SAVED_BLEND', out_blend)
print('SAVED_OBJ', out_obj)
