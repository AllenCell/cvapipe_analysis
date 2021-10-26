import vtk
import math
import operator
import numpy as np
from functools import reduce
from aicsshparam import shtools
import matplotlib.pyplot as plt
from matplotlib import animation
from vtk.util import numpy_support as vtknp

class MeshToolKit():

    @staticmethod
    def get_mesh_from_series(row, alias, lmax):
        coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
        for l in range(lmax):
            for m in range(l + 1):
                try:
                    # Cosine SHE coefficients
                    coeffs[0, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                    ]
                    # Sine SHE coefficients
                    coeffs[1, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                    ]
                # If a given (l,m) pair is not found, it is assumed to be zero
                except: pass
        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
        return mesh

    @staticmethod
    def find_plane_mesh_intersection(mesh, proj, use_vtk_for_intersection=True):

        # Find axis orthogonal to the projection of interest
        axis = [a for a in [0, 1, 2] if a not in proj][0]

        # Get all mesh points
        points = vtknp.vtk_to_numpy(mesh.GetPoints().GetData())

        if not np.abs(points[:, axis]).sum():
            raise Exception("Only zeros found in the plane axis.")

        if use_vtk_for_intersection:

            mid = np.mean(points[:, axis])
            '''Set the plane a little off center to avoid undefined intersections.
            Without this the code hangs when the mesh has any edge aligned with the
            projection plane. Also add a little of noisy to the coordinates to
            help with the same problem.'''
            mid += 0.75
            offset = 0.1 * np.ptp(points, axis=0).max()

            # Create a vtkPlaneSource
            plane = vtk.vtkPlaneSource()
            plane.SetXResolution(4)
            plane.SetYResolution(4)
            if axis == 0:
                plane.SetOrigin(
                    mid, points[:, 1].min() - offset, points[:, 2].min() - offset
                )
                plane.SetPoint1(
                    mid, points[:, 1].min() - offset, points[:, 2].max() + offset
                )
                plane.SetPoint2(
                    mid, points[:, 1].max() + offset, points[:, 2].min() - offset
                )
            if axis == 1:
                plane.SetOrigin(
                    points[:, 0].min() - offset, mid, points[:, 2].min() - offset
                )
                plane.SetPoint1(
                    points[:, 0].min() - offset, mid, points[:, 2].max() + offset
                )
                plane.SetPoint2(
                    points[:, 0].max() + offset, mid, points[:, 2].min() - offset
                )
            if axis == 2:
                plane.SetOrigin(
                    points[:, 0].min() - offset, points[:, 1].min() - offset, mid
                )
                plane.SetPoint1(
                    points[:, 0].min() - offset, points[:, 1].max() + offset, mid
                )
                plane.SetPoint2(
                    points[:, 0].max() + offset, points[:, 1].min() - offset, mid
                )
            plane.Update()
            plane = plane.GetOutput()

            # Trangulate the plane
            triangulate = vtk.vtkTriangleFilter()
            triangulate.SetInputData(plane)
            triangulate.Update()
            plane = triangulate.GetOutput()

            # Calculate intersection
            intersection = vtk.vtkIntersectionPolyDataFilter()
            intersection.SetInputData(0, mesh)
            intersection.SetInputData(1, plane)
            intersection.Update()
            intersection = intersection.GetOutput()

            # Get coordinates of intersecting points
            points = vtknp.vtk_to_numpy(intersection.GetPoints().GetData())
            coords = points[:, proj]

        else:
            
            valids = np.where((points[:,axis] > -2.5)&(points[:,axis] < 2.5))
            coords = points[valids[0]][:,proj]

        # Sorting points clockwise
        # This has been discussed here:
        # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
        # but seems not to be very efficient. Better version is proposed here:
        # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
        center = tuple(
            map(
                operator.truediv,
                reduce(lambda x, y: map(operator.add, x, y), coords),
                [len(coords)] * 2,
            )
        )
        coords = sorted(
            coords,
            key=lambda coord: (
                -135
                - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
                )
            )
            % 360,
        )

        # Store sorted coordinates
        # points[:, proj] = coords
        return np.array(coords)

    @staticmethod
    def sort_2d_points(coords):
        # Sorting points clockwise
        # This has been discussed here:
        # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
        # but seems not to be very efficient. Better version is proposed here:
        # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
        center = tuple(
            map(
                operator.truediv,
                reduce(lambda x, y: map(operator.add, x, y), coords),
                [len(coords)] * 2,
            )
        )
        coords = sorted(
            coords,
            key=lambda coord: (
                -135
                - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
                )
            )
            % 360,
        )

        # Store sorted coordinates
        # points[:, proj] = coords
        return np.array(coords)

    @staticmethod
    def get_2d_contours(named_meshes, swapxy_on_zproj=False):
        contours = {}
        projs = [[0, 1], [0, 2], [1, 2]]
        if swapxy_on_zproj:
            projs = [[0, 1], [1, 2], [0, 2]]
        for dim, proj in zip(["z", "y", "x"], projs):
            contours[dim] = {}
            for alias, meshes in named_meshes.items():
                contours[dim][alias] = []
                for mesh in meshes:
                    coords = MeshToolKit.find_plane_mesh_intersection(mesh, proj)
                    if swapxy_on_zproj and dim == 'z':
                        coords = coords[:, ::-1]
                    contours[dim][alias].append(coords)
        return contours

    @staticmethod
    def animate_contours(control, contours, save=None):
        hmin, hmax, vmin, vmax = control.get_plot_limits()
        offset = 0.05 * (hmax - hmin)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.tight_layout()
        plt.close()
        ax.set_xlim(hmin - offset, hmax + offset)
        ax.set_ylim(vmin - offset, vmax + offset)
        ax.set_aspect("equal")
        if not control.get_plot_frame():
            ax.axis("off")

        lines = []
        for alias, _ in contours.items():
            color = control.get_color_from_alias(alias)
            (line,) = ax.plot([], [], lw=2, color=color)
            lines.append(line)

        def animate(i):
            for alias, line in zip(contours.keys(), lines):
                ct = contours[alias][i]
                mx = ct[:, 0]
                my = ct[:, 1]
                line.set_data(mx, my)
            return lines
        
        n = len(contours[list(contours.keys())[0]])
        anim = animation.FuncAnimation(
            fig, animate, frames=n, interval=100, blit=True
        )
        if save is not None:
            anim.save(save, fps=n)
            plt.close("all")
            return
        return anim
