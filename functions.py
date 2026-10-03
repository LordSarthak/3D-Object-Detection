from PyQt5.QtWidgets import QColorDialog, QMessageBox
from PyQt5 import QtWidgets
from OpenGL.GLUT import *
import open3d as o3d
import os
import sys
import re
import numpy as np


def open_file_ask(opengl_obj, obj_path, obj_name, uv_label, material_label, drawcalls_label, vertices_label, triangles_label, edges_label):
    file_name = QtWidgets.QFileDialog.getOpenFileName(
        None, 'Open file', '', "Mesh files (*.obj *.stl *.ply *.off)")
    if not file_name[0]:
        return
    open_file(file_name[0], opengl_obj, obj_path, obj_name, uv_label, material_label, drawcalls_label, vertices_label, triangles_label, edges_label)


def open_file(file_name, opengl_obj, obj_path, obj_name, uv_label, material_label, drawcalls_label, vertices_label, triangles_label, edges_label):
    mesh = o3d.io.read_triangle_mesh(file_name)
    if not mesh.has_triangles():
        print("Error: Mesh does not contain triangles.")
        return

    mesh.compute_vertex_normals()
    opengl_obj.set_mesh(mesh)  # Make sure your OpenGL object can handle Open3D meshes
    set_name(file_name, obj_path, obj_name)
    set_file_info(mesh, vertices_label, triangles_label, edges_label)

    _, file_extension = os.path.splitext(file_name)
    file_format = file_extension.replace(".", "").lower()
    if file_format == "obj":
        has_uv(file_name, uv_label)
        materials(file_name, material_label)
        draw_calls(file_name, drawcalls_label)


def set_name(file_name, obj_path, obj_name):
    file_path = os.path.normpath(file_name)
    ob_name = os.path.basename(file_name)
    obj_path.setText(file_path)
    obj_name.setText(ob_name)


def set_file_info(mesh, vertices_label, triangles_label, edges_label):
    vertex_count = len(mesh.vertices)
    triangle_count = len(mesh.triangles)
    edge_count = count_edges(mesh)
    vertices_label.setText(str(vertex_count))
    triangles_label.setText(str(triangle_count))
    edges_label.setText(str(edge_count))


def count_edges(mesh):
    edge_set = set()
    triangles = np.asarray(mesh.triangles)
    for tri in triangles:
        i, j, k = tri
        edge_set.add(tuple(sorted((i, j))))
        edge_set.add(tuple(sorted((j, k))))
        edge_set.add(tuple(sorted((k, i))))
    return len(edge_set)


def has_uv(file_path, has_label):
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("vt "):
                    has_label.setText("Yes")
                    return
        has_label.setText("No")
    except Exception:
        has_label.setText("Error")


def materials(file_path, material_label):
    try:
        with open(file_path, "r") as file:
            contents = file.read()
            num = set(re.findall(r'usemtl (\S+)', contents))
            material_label.setText(str(len(num)))
    except Exception as e:
        material_label.setText("Error")
        print(f"Material parsing error: {e}")


def draw_calls(file_path, drawcalls_label):
    try:
        with open(file_path, "r") as file:
            contents = file.read()
            draw = re.findall(r'^usemtl (\S+)', contents, re.MULTILINE)
            drawcalls_label.setText(str(len(draw)) if draw else "0")
    except Exception:
        drawcalls_label.setText("Error")


def get_color(button, button_color, btn_name, openGL):
    color_dialog = QColorDialog()
    color = color_dialog.getColor()
    if color.isValid():
        r, g, b, a = color.getRgb()
        color = f"rgb({r}, {g}, {b})"
        button_color(button, color)
        if btn_name == "background":
            openGL.background_color((r / 255, g / 255, b / 255))
        if btn_name == "wire":
            openGL.change_light_color((r / 255, g / 255, b / 255))


def set_button_color(button, color):
    button.setStyleSheet(f"background-color: {color};"
                         f"border-radius: 2px;"
                         )


def change_slider(slider, line, openGL, btn_name=""):
    value = slider.value()
    line.setText(str(value))
    if btn_name == "fov":
        openGL.update_fov(value)
    elif btn_name == "wireframe":
        openGL.update_alpha(value)
    elif btn_name == "grid":
        openGL.update_grid_alpha(value)


def update_slider(slider, line):
    value = int(line.text())
    slider.setValue(value)


def update_grid_size(text, openGL, btn_name):
    value = int(text.text())
    if btn_name == "cell":
        openGL.update_grid_cell(value)
    elif btn_name == "size":
        openGL.update_grid_size(value)


def close_file(openGL, obj_path, obj_name):
    openGL.set_mesh(None)
    obj_path.setText("")
    obj_name.setText("")


def show_message_box():
    title = "About Qt 3DViewer"
    message = (
        "Qt 3DViewer is a compact tool for \n"
        "viewing 3D models in a user friendly way\n"
        "Designed and Developed by Krrish.\n\n"
        "Powered by: Python 3.9, Qt Designer,\n"
        "PyQt5, OpenGL, Open3D and ModernGL.\n\n"
        "Movement:\n"
        "•  Left mouse button to rotate\n"
        "•  Mouse wheel to zoom\n"
        "•  Right mouse button to pan."
    )
    msg = QMessageBox()
    msg.setStyleSheet("""
                      QMessageBox {
                          background-color: rgb(56, 56, 56);
                          font-size: 14px;
                          color: rgb(145, 145, 145);
                      }
                      """)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setStandardButtons(QMessageBox.Close)
    msg.exec_()


def exit_app():
    QtWidgets.QApplication.quit()
