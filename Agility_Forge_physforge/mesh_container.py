import meshio
import gmsh
import json
import math
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from pathlib import Path
import os


crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, "data")
msh_dir = os.path.join(data_dir, "msh/jax_forge")


class MeshContainer:
    """
    This class holds the references to a surface and volumetric mesh. These two meshes represent the
    same geometry and will always be in sync. When creating an instance, you can supply a json of
    either the surface or volumetric mesh. Conversely, you can also supply meshio representation of
    either. Default for meshio representation is for a volumetric mesh.

    If you supply a meshio representation, make sure to get "is_json" to False.
    If you supply a surface mesh, make sure to get "is_surface" to True.

    Important: The default be

    Surface mesh json format:
    {faceface
        "Vertices": [float],
        "Triangles": [int]
    }

    Volumetric mesh json format:
    {
        "points": [float],
        "cells": [int]
    }
    """

    def __init__(self, mesh, is_json=True, is_surface=False, standalone=False,tmp_filepath=None):
        self.tmp_filepath = Path(tmp_filepath) if tmp_filepath else msh_dir / Path("tmp")
        self.tmp_obj = self.tmp_filepath / "tmp.obj"
        self.tmp_stl = self.tmp_filepath / "tmp.stl"
        self.tmp_vtk = self.tmp_filepath / "tmp.vtk"
        self.test_obj = self.tmp_filepath / "test.obj"
        self.test_stl = self.tmp_filepath / "test.stl"
        self.test_vtk = self.tmp_filepath / "test.vtk"

        if is_surface:
            if is_json:
                self.__deserialize_obj(mesh)
            else:
                self.obj = mesh
            if not standalone:
                self.get_volumetric()
        else:
            if is_json:
                self.__deserialize_vtk(mesh)
            else:
                self.vtk = mesh
            if not standalone:
                self.get_surface(True)
   
    @classmethod
    def from_obj(cls, mesh_path, standalone=False):
        """A helper method for converting surface obj representations to MeshContainers"""
        with open(mesh_path, 'r') as file:
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            mesh = cls(meshio.read(file, file_format='obj'), 
                    is_surface=True, 
                    is_json=False,
                    standalone=standalone)
            gmsh.finalize()
        return mesh
    
    @classmethod 
    def from_json(cls, mesh_path, standalone=False):
        """A helper method for converting surface json representations to MeshContainers"""
        with open(mesh_path, 'r') as file:
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            mesh = cls(json.load(file), 
                    is_surface=True, 
                    is_json=True,
                    standalone=standalone)
            gmsh.finalize()
        return mesh
    
    @classmethod 
    def from_db(cls, vertices: list, triangles: list, tmp_filepath = None):
        """A helper method for converting creating MeshContainers directly from the DB JSON buffer data"""
        # If tmp_filepath is not provided, use a default path
        if tmp_filepath is None:
            tmp_filepath = msh_dir / "tmp"  # Default path, can be adjusted
        
        mesh = MeshContainer(
            {"Vertices": vertices, "Triangles": triangles},
            is_surface=True,
            standalone=True,
            tmp_filepath=tmp_filepath
        )        
        return mesh

    def __deserialize_obj(self, json_data):
        with open(self.tmp_obj, "w") as file:

            file.write(
                "\n".join(
                    [
                        f"v {vertex[0]} {vertex[1]} {vertex[2]}"
                        for vertex in [
                            json_data["Vertices"][i: i + 3]
                            for i in range(0, len(json_data["Vertices"]), 3)
                        ]
                    ]
                )
            )
            file.write("\n")
            file.write(
                "\n".join(
                    [
                        f"f {face[0]+1} {face[1]+1} {face[2]+1}"
                        for face in [
                            json_data["Triangles"][i: i + 3]
                            for i in range(0, len(json_data["Triangles"]), 3)
                        ]
                    ]
                )
            )

        self.obj = meshio.read(self.tmp_obj)

    def __deserialize_vtk(self, json):
        self.vtk = meshio.Mesh(json["points"], {"tetra": json["cells"]})

    def __prune_unused_points(self, input_points, input_faces):
        """
        Removes unused points from the mesh and updates face indices accordingly.

        Args:
            input_points: numpy array of points (n_points, 3)
            input_faces: numpy array of face indices (n_faces, 3)

        Returns:
            tuple: (pruned_points, pruned_faces)
        """
        original_shape = input_faces.shape
        used_point_indices = input_faces.flatten()
        unique_indices, inverse_mapping = np.unique(used_point_indices, return_inverse=True)
        pruned_points = input_points[unique_indices]
        pruned_faces = inverse_mapping.reshape(original_shape)

        return pruned_points, pruned_faces



    def get_surface(self, do_not_flip=False):
        """
        Removes internal points from a volumetric mesh and returns the surface mesh.

        Args:
            do_not_flip: If True, the normals of the faces will not be flipped.
            (Important for converting between Left handed and Right handed coordinate systems)
        """
        surface_faces = self.extract_faces()
        pruned_points, pruned_faces = self.__prune_unused_points(self.vtk.points, surface_faces)
        self.obj = meshio.Mesh(points=pruned_points, cells=[("triangle", pruned_faces)])
    
    def get_metrics(self):
        # Calculate quality metrics
        elements = self.vtk.cells_dict["tetra"]
        points = self.vtk.points
        volumes = []
        aspect_ratios = []

        for tet in elements:
            verts = points[tet]
            edges = verts[1:] - verts[0]
            volume = abs(np.dot(edges[0], np.cross(edges[1], edges[2]))) / 6
            volumes.append(volume)

            edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
            max_edge = np.max(edge_lengths)
            min_edge = np.min(edge_lengths)
            if min_edge > 0:
                aspect_ratio = max_edge / min_edge
                aspect_ratios.append(aspect_ratio)

        stats = {
            "num_elements": len(elements),
            "avg_aspect_ratio": float(np.mean(aspect_ratios)),
            "max_aspect_ratio": float(np.max(aspect_ratios)),
            "volume_ratio": float(np.max(volumes) / np.min(volumes)),
        }

        print("\nMesh Quality Metrics:")
        print(f"Number of elements: {stats['num_elements']}")
        print(f"Average aspect ratio: {stats['avg_aspect_ratio']:.3f}")
        print(f"Maximum aspect ratio: {stats['max_aspect_ratio']:.3f}")
        print(f"Volume ratio (max/min): {stats['volume_ratio']:.3f}")

        return stats

    def compute_tet_aspect_ratios(self):
        pts = self.vtk.points
        elems = self.vtk.cells_dict["tetra"]
        p = pts[elems]
        e = np.linalg.norm(np.stack([
            p[:,0]-p[:,1], p[:,0]-p[:,2], p[:,0]-p[:,3],
            p[:,1]-p[:,2], p[:,1]-p[:,3], p[:,2]-p[:,3]
        ], 1), axis=2)
        le = e.max(1)
        v = np.abs(np.einsum('ij,ij->i', p[:,0]-p[:,3], np.cross(p[:,1]-p[:,3], p[:,2]-p[:,3]))) / 6
        A = lambda a,b,c: 0.5 * np.linalg.norm(np.cross(b-a, c-a), axis=1)
        h = lambda a: 3*v / (A(*a)+1e-12)
        ha = np.stack([
            h((p[:,1],p[:,2],p[:,3])),
            h((p[:,0],p[:,2],p[:,3])),
            h((p[:,0],p[:,1],p[:,3])),
            h((p[:,0],p[:,1],p[:,2]))
        ], axis=1).min(1)
        ar_np = le / (ha + 1e-12)
        return np.max(ar_np), np.mean(ar_np)

    def get_volumetric(self, mesh_density=2.5, refinement_factor=1.0, force_delaunay=False) -> "dict":
        stl_path = str(self.tmp_stl)
        vtk_path = str(self.tmp_vtk)
        self.obj.write(stl_path)
        gmsh.open(stl_path)

        bounds = gmsh.model.getBoundingBox(-1, -1)
        model_size = max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
        print(f"Model size: {model_size}")

        angle = 40
        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180,
            1,  # includeBoundary
            1,  # forceParametrizablePatches
            180 * math.pi / 180
        )
        gmsh.model.mesh.createGeometry()
        surfaces = gmsh.model.getEntities(2)
        surf_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
        gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", [s[1] for s in surfaces])
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        base_size = mesh_density / refinement_factor
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", base_size / 2)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", base_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", model_size * 0.01)
        gmsh.model.mesh.field.setNumber(2, "DistMax", model_size * 0.1)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        if force_delaunay:
            print("Using Delaunay mesher...")
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        else:
            print("Using Netgen mesher...")
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)

        gmsh.model.mesh.generate(3)

        for _ in range(3):
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.optimize("Gmsh")
            gmsh.model.mesh.optimize("HighOrder")

        gmsh.write(vtk_path)
        self.vtk = meshio.read(vtk_path)
        return  # optionally call self.get_metrics()

    def extract_faces(self) -> "np.ndarray":
        """
        Extracts triangles with consistent winding order that make up the 
        surface of the volumetric mesh. Assumes standard tetrahedron winding 
        resulting in outward-pointing normals.

        Returns:
            np.ndarray: A numpy array (N, 3) containing surface triangle 
                        definitions with consistent winding.
        """
        tets = self.vtk.get_cells_type("tetra")
        if not tets.size:
            return np.empty((0, 3), dtype=tets.dtype)

        # Define faces for each tet with outward-pointing normal convention
        original_faces = np.vstack([
            tets[:, [1, 3, 2]],  # Opposing v0
            tets[:, [0, 2, 3]],  # Opposing v1
            tets[:, [0, 1, 2]],  # Opposing v2
            tets[:, [0, 3, 1]]   # Opposing v3
        ])

        # Create canonical representation (sorted) for identification
        sorted_faces = np.sort(original_faces, axis=1)
        sorted_faces = [tuple(face) for face in sorted_faces]

        # Count occurrences of canonical faces
        face_counts = Counter(sorted_faces)

        # Identify unique canonical faces (surface faces)
        unique_sorted_faces = {face for face, count in face_counts.items() if count == 1}

        # Retrieve the original oriented faces corresponding to the unique ones
        triangle_mask = np.array([face_tuple in unique_sorted_faces for face_tuple in sorted_faces])
        outer_faces = original_faces[triangle_mask]

        return outer_faces


    def write_vtk(self):
        """
        Writes the volumetric mesh to a vtk file.
        """
        meshio.write(self.test_vtk, self.vtk)

    def write_obj(self):
        """
        Writes the surface mesh to an obj file.
        """
        self.obj.write(self.test_obj)

    def serialize_vtk(self) -> "dict":
        """
        Creates a dictionary representation of the volumetric mesh.

        Returns:
            mesh_data: A dictionary representation of the volumetric mesh
        """
        points = self.vtk.points
        cells = self.vtk.cells_dict["tetra"]
        mesh_data = {"points": points.tolist(), "cells": cells.tolist()}
        return mesh_data


    def serialize_obj(self):
        """
        Creates a dictionary representation of the surface mesh.

        Returns:
            mesh_data: A dictionary representation of the surface mesh
        """

        self.obj.write(self.tmp_obj)
        vertices = []
        triangles = []
        with open(self.tmp_obj, "r") as file:

            for line in file:
                if line[0] == "v":
                    vertices.extend([float(i) for i in line.split()[1:]])
                elif line[0] == "f":
                    triangles.extend([int(i) - 1 for i in line.split()[1:]])



        mesh_data = {
            "Vertices": vertices,
            "Triangles": triangles,
        }

        return mesh_data
