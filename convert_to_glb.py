#!/usr/bin/env python3
"""Converts Blender-exported USDC files to GLB for use in three.js / model-viewer."""

import sys
import struct
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf
import pygltflib


def triangulate(face_vertex_counts, face_vertex_indices):
    """Fan-triangulate n-gons to triangles."""
    tris = []
    idx = 0
    for count in face_vertex_counts:
        verts = face_vertex_indices[idx:idx + count]
        for i in range(1, count - 1):
            tris.extend([verts[0], verts[i], verts[i + 1]])
        idx += count
    return tris


def get_material_color(prim):
    """Extract diffuseColor and metallic/roughness from UsdPreviewSurface binding."""
    binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
    if not binding:
        return [0.8, 0.8, 0.8], 0.0, 0.5, 1.0

    for shader_prim in binding.GetPrim().GetChildren():
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            continue
        diffuse_input = shader.GetInput("diffuseColor")
        metallic_input = shader.GetInput("metallic")
        roughness_input = shader.GetInput("roughness")
        opacity_input = shader.GetInput("opacity")

        color = [0.8, 0.8, 0.8]
        metallic = 0.0
        roughness = 0.5
        opacity = 1.0

        if diffuse_input:
            v = diffuse_input.Get()
            if v:
                color = list(v)
        if metallic_input:
            v = metallic_input.Get()
            if v is not None:
                metallic = float(v)
        if roughness_input:
            v = roughness_input.Get()
            if v is not None:
                roughness = float(v)
        if opacity_input:
            v = opacity_input.Get()
            if v is not None:
                opacity = float(v)
        return color, metallic, roughness, opacity

    return [0.8, 0.8, 0.8], 0.0, 0.5, 1.0


def expand_facevarying(values, face_vertex_counts, face_vertex_indices):
    """Expand faceVarying attribute to per-vertex-occurrence (unshared)."""
    result = []
    for v in values:
        result.append(list(v) if hasattr(v, '__iter__') else v)
    return result


def usdc_to_glb(input_path, output_path):
    print(f"\nKonvertiere: {input_path}")
    stage = Usd.Stage.Open(input_path)

    gltf = pygltflib.GLTF2()
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[0])]
    root_node = pygltflib.Node(name="root", children=[])
    gltf.nodes = [root_node]

    binary_data = bytearray()
    materials_map = {}  # prim_path → material index

    mesh_node_indices = []

    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Mesh':
            continue

        mesh_prim = UsdGeom.Mesh(prim)

        # --- Geometry ---
        points = mesh_prim.GetPointsAttr().Get()
        if points is None:
            continue
        points = [list(p) for p in points]

        face_counts = list(mesh_prim.GetFaceVertexCountsAttr().Get())
        face_indices = list(mesh_prim.GetFaceVertexIndicesAttr().Get())
        normals_attr = mesh_prim.GetNormalsAttr().Get()
        normals_interp = mesh_prim.GetNormalsInterpolation()

        # --- Triangulate ---
        tri_indices = triangulate(face_counts, face_indices)

        # Handle faceVarying normals: expand positions to per-face-vertex
        if normals_attr and normals_interp == UsdGeom.Tokens.faceVarying:
            # Build expanded arrays (no vertex sharing, one entry per face-vertex)
            normals_list = [list(n) for n in normals_attr]

            # Map from face-vertex index to expanded position
            exp_positions = []
            exp_normals = []
            exp_tri_indices = []

            fv_to_exp = {}  # (face_vertex_occurrence_idx) → exp_idx
            fv_idx = 0
            for fi, count in enumerate(face_counts):
                for vi in range(count):
                    orig_vert = face_indices[fv_idx]
                    pos = points[orig_vert]
                    norm = normals_list[fv_idx]
                    exp_idx = len(exp_positions)
                    fv_to_exp[fv_idx] = exp_idx
                    exp_positions.append(pos)
                    exp_normals.append(norm)
                    fv_idx += 1

            # Re-triangulate using expanded indices
            fv_idx = 0
            exp_tri_indices = []
            for fi, count in enumerate(face_counts):
                base = fv_idx
                fv_verts = list(range(base, base + count))
                for i in range(1, count - 1):
                    exp_tri_indices.extend([fv_verts[0], fv_verts[i], fv_verts[i + 1]])
                fv_idx += count

            final_positions = exp_positions
            final_normals = exp_normals
            final_indices = exp_tri_indices
        else:
            # vertex interpolation or no normals
            final_positions = points
            if normals_attr:
                final_normals = [list(n) for n in normals_attr]
                if len(final_normals) != len(final_positions):
                    final_normals = None
            else:
                final_normals = None
            final_indices = tri_indices

        if not final_indices:
            continue

        # --- Flip Z-up to Y-up (Blender → GLTF) ---
        # GLTF is Y-up, Blender/USD is Z-up
        # Transform: (x, y, z) → (x, z, -y)
        positions_yup = [[p[0], p[2], -p[1]] for p in final_positions]
        if final_normals:
            normals_yup = [[n[0], n[2], -n[1]] for n in final_normals]
        else:
            normals_yup = None

        # --- Pack binary data ---
        pos_array = np.array(positions_yup, dtype=np.float32)
        idx_array = np.array(final_indices, dtype=np.uint32)

        # Align to 4 bytes
        def align4(data):
            pad = (4 - len(data) % 4) % 4
            return bytes(data) + b'\x00' * pad

        # Positions buffer view
        pos_bytes = pos_array.tobytes()
        pos_offset = len(binary_data)
        binary_data.extend(align4(pos_bytes))

        pos_bv = pygltflib.BufferView(
            buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes),
            target=pygltflib.ARRAY_BUFFER
        )
        pos_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pos_bv)

        pos_min = pos_array.min(axis=0).tolist()
        pos_max = pos_array.max(axis=0).tolist()
        pos_acc = pygltflib.Accessor(
            bufferView=pos_bv_idx, componentType=pygltflib.FLOAT,
            count=len(positions_yup), type=pygltflib.VEC3,
            min=pos_min, max=pos_max
        )
        pos_acc_idx = len(gltf.accessors)
        gltf.accessors.append(pos_acc)

        # Normals buffer view
        norm_acc_idx = None
        if normals_yup:
            norm_array = np.array(normals_yup, dtype=np.float32)
            norm_bytes = norm_array.tobytes()
            norm_offset = len(binary_data)
            binary_data.extend(align4(norm_bytes))

            norm_bv = pygltflib.BufferView(
                buffer=0, byteOffset=norm_offset, byteLength=len(norm_bytes),
                target=pygltflib.ARRAY_BUFFER
            )
            norm_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(norm_bv)

            norm_acc = pygltflib.Accessor(
                bufferView=norm_bv_idx, componentType=pygltflib.FLOAT,
                count=len(normals_yup), type=pygltflib.VEC3
            )
            norm_acc_idx = len(gltf.accessors)
            gltf.accessors.append(norm_acc)

        # Indices buffer view
        idx_bytes = idx_array.tobytes()
        idx_offset = len(binary_data)
        binary_data.extend(align4(idx_bytes))

        idx_bv = pygltflib.BufferView(
            buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER
        )
        idx_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(idx_bv)

        idx_acc = pygltflib.Accessor(
            bufferView=idx_bv_idx, componentType=pygltflib.UNSIGNED_INT,
            count=len(final_indices), type=pygltflib.SCALAR
        )
        idx_acc_idx = len(gltf.accessors)
        gltf.accessors.append(idx_acc)

        # --- Material ---
        mat_path = str(UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath())
        if mat_path not in materials_map:
            color, metallic, roughness, opacity = get_material_color(prim)
            mat = pygltflib.Material(
                name=mat_path,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=color + [opacity],
                    metallicFactor=metallic,
                    roughnessFactor=roughness
                ),
                doubleSided=True,
                alphaMode="MASK" if opacity < 1.0 else "OPAQUE"
            )
            mat_idx = len(gltf.materials)
            gltf.materials.append(mat)
            materials_map[mat_path] = mat_idx
        else:
            mat_idx = materials_map[mat_path]

        # --- Mesh + Node ---
        attrs = pygltflib.Attributes(POSITION=pos_acc_idx)
        if norm_acc_idx is not None:
            attrs.NORMAL = norm_acc_idx

        primitive = pygltflib.Primitive(
            attributes=attrs,
            indices=idx_acc_idx,
            material=mat_idx,
            mode=pygltflib.TRIANGLES
        )
        mesh = pygltflib.Mesh(name=prim.GetName(), primitives=[primitive])
        mesh_idx = len(gltf.meshes)
        gltf.meshes.append(mesh)

        node = pygltflib.Node(name=prim.GetName(), mesh=mesh_idx)
        node_idx = len(gltf.nodes)
        gltf.nodes.append(node)
        root_node.children.append(node_idx)

        print(f"  Mesh: {prim.GetName()} → {len(final_positions)} vertices, {len(final_indices)//3} triangles")

    # --- Buffer ---
    gltf.buffers = [pygltflib.Buffer(byteLength=len(binary_data))]
    gltf.set_binary_blob(bytes(binary_data))

    gltf.save(output_path)
    import os
    size = os.path.getsize(output_path)
    print(f"  → {output_path} ({size//1024} KB)")


if __name__ == '__main__':
    import os

    models_dir = '/Users/marc/development/3d-database/models/'
    files = [
        ('waermepumpe_wolf.usdc', 'waermepumpe_wolf.glb'),
        ('waermepumpe_wolf_mit_bodenkonsole.usdc', 'waermepumpe_wolf_mit_bodenkonsole.glb'),
        ('waermepumpe_wolf_bodenkonsole_abstaende.usdc', 'waermepumpe_wolf_bodenkonsole_abstaende.glb'),
    ]

    for src, dst in files:
        usdc_to_glb(models_dir + src, models_dir + dst)

    print('\nAlle Konvertierungen abgeschlossen!')
