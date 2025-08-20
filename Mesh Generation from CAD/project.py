import numpy as np
import sys
import os
import gmsh # Modification 
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QComboBox
)
import pyvista as pv
from pyvistaqt import QtInteractor
import meshio
class MeshApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD to Mesh Generator")
        self.resize(1000, 700)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Status Label
        self.label = QLabel("No CAD file uploaded")
        self.layout.addWidget(self.label)

        # Upload Button
        self.upload_btn = QPushButton("Upload CAD File (STEP/IGES)")
        self.upload_btn.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_btn)

        # Mesh dimension selector
        self.dimension_selector = QComboBox()
        self.dimension_selector.addItems(["1D (Lines)", "2D (Surfaces)", "3D (Volumes)"])
        self.layout.addWidget(self.dimension_selector)

        # Generate Mesh Button
        self.mesh_btn = QPushButton("Generate Mesh")
        self.mesh_btn.clicked.connect(self.generate_mesh)
        self.layout.addWidget(self.mesh_btn)

        # PyVista 3D Plot widget
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        self.cad_file = None
        self.mesh_file = None  # filename set dynamically
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CAD file", "",
            "STEP Files (*.step *.stp);;IGES Files (*.iges *.igs)"
        )
        if file_path:
            self.cad_file = file_path
            self.label.setText(f"📂 Uploaded: {os.path.basename(file_path)}")
            print(f"Uploaded file: {file_path}")
    def generate_mesh(self):
        if not self.cad_file:
            self.label.setText("⚠️ Please upload a CAD file first!")
            return
        try:
            gmsh.initialize()
            gmsh.model.add("CAD_Mesh")

            # Import CAD geometry
            gmsh.model.occ.importShapes(self.cad_file)
            gmsh.model.occ.synchronize()

            # Mesh Settings
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)

            # Get selected dimension
            dim_choice = self.dimension_selector.currentIndex() + 1  # 1=1D, 2=2D, 3=3D

            # Choose output filename
            if dim_choice == 1:
                self.mesh_file = "1d_output.vtk"
            elif dim_choice == 2:
                self.mesh_file = "2d_output.vtk"
            elif dim_choice == 3:
                self.mesh_file = "3d_output.vtk"

            gmsh.model.mesh.generate(dim_choice)

            # Save mesh
            gmsh.write(self.mesh_file)
            gmsh.finalize()

            self.label.setText(f"✅ {dim_choice}D Mesh generated: {self.mesh_file}")
            print(f"Mesh saved to {self.mesh_file}")

            # Load mesh with meshio
            mesh_data = meshio.read(self.mesh_file)
            points = mesh_data.points

            pv_mesh = None

            if dim_choice == 3 and "tetra" in mesh_data.cells_dict:
                # tetrahedral cells for 3D
                tetra_cells = mesh_data.cells_dict["tetra"]
                vtk_cells = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).astype(np.int64)
                pv_mesh = pv.UnstructuredGrid(vtk_cells, np.full(len(tetra_cells), 10), points)  # VTK_TETRA=10

            elif dim_choice == 2 and "triangle" in mesh_data.cells_dict:
                # triangles for 2D
                tri_cells = mesh_data.cells_dict["triangle"]
                faces = np.hstack([np.full((tri_cells.shape[0], 1), 3), tri_cells]).astype(np.int64)
                pv_mesh = pv.PolyData(points, faces)

            elif dim_choice == 1 and "line" in mesh_data.cells_dict:
                # lines for 1D
                line_cells = mesh_data.cells_dict["line"]
                lines = np.hstack([np.full((line_cells.shape[0], 1), 2), line_cells]).astype(np.int64)
                pv_mesh = pv.PolyData(points, lines=lines)

            else:
                # fallback
                pv_mesh = pv.read(self.mesh_file)

            # Visualize
            self.plotter.clear()

            if dim_choice == 1:
                # 1D mesh → black thick lines
                self.plotter.add_mesh(pv_mesh, color="black", line_width=3)

            elif dim_choice == 2:
                # 2D mesh → light blue with dark edges
                self.plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True, edge_color="black")

            elif dim_choice == 3:
                # 3D mesh → translucent + slice to see inside
                self.plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True, opacity=0.4)
                slice_mesh = pv_mesh.slice(normal="z")
                self.plotter.add_mesh(slice_mesh, color="red", show_edges=True)

            self.plotter.reset_camera()

        except Exception as e:
            self.label.setText(f"❌ Error generating mesh: {e}")
            print("Error:", e)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshApp()
    window.show()
    sys.exit(app.exec_())
