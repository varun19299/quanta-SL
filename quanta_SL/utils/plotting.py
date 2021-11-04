from pathlib import Path

import cv2
import numpy as np
from einops import repeat
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict
import open3d


def save_plot(savefig: bool = False, show: bool = True, **kwargs) -> object:
    """
    Helper function for saving plots
    :param savefig: Whether to save the figure
    :param show: Display in graphical window or just close the plot
    :param kwargs: fname, close
    :return:
    """
    if "close" in kwargs:
        close = kwargs["close"]
    else:
        close = not show

    if savefig:
        path = Path(kwargs["fname"])
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(kwargs["fname"], dpi=150, bbox_inches="tight", transparent=True)

    if show:
        plt.show()
    if close:
        plt.close()


def subplot_extent(ax, fig, pad=0.0):
    """
    Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad).transformed(
        fig.dpi_scale_trans.inverted()
    )


def arrowed_spines(
    ax,
    x_width_fraction=0.05,
    x_height_fraction=0.05,
    lw=None,
    ohg=0.3,
    locations=("bottom right", "left up"),
    remove_spine: bool = False,
    **arrow_kwargs,
):
    """
    Add arrows to the requested spines
    Code originally sourced here: https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. If not supplied, default behaviour is to use the value on the current left spine.
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". These must be orthogonal
    (e.g. "left left" will result in an error). Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault("overhang", ohg)
    arrow_kwargs.setdefault("clip_on", False)
    arrow_kwargs.update({"length_includes_head": True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines["left"].get_linewidth()

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if remove_spine:
        # removing the default axis on all sides:
        for side in ["bottom", "right", "top", "left"]:
            ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax - ymin)
    hl = x_height_fraction * (xmax - xmin)

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(" ")
        assert side in {"top", "bottom", "left", "right"}, "Unsupported side"
        assert direction in {"up", "down", "left", "right"}, "Unsupported direction"

        if side in {"bottom", "top"}:
            if direction in {"up", "down"}:
                raise ValueError(
                    "Only left/right arrows supported on the bottom and top"
                )

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == "bottom" else ymax

            if direction == "right":
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {"left", "right"}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == "left" else xmax

            if direction == "up":
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax

        # arrow_kwargs.update("")

        annots[loc_str] = ax.arrow(
            x,
            y,
            dx,
            dy,
            # fc="k",
            # ec="k",
            lw=lw,
            head_width=head_width,
            head_length=head_length,
            **arrow_kwargs,
        )

    return annots


def plot_code_LUT(
    code_LUT: np.ndarray, show: bool = True, aspect_ratio: float = 3.0, **kwargs
):
    """
    Image illustrating coding scheme
    :param code_LUT: Code Look-Up-Table
    """
    h, c = code_LUT.shape

    num_repeat = kwargs.get("num_repeat", max(int(h / c / aspect_ratio), 1))

    code_img = repeat(code_LUT, "h c -> (c repeat) h", repeat=num_repeat)

    if kwargs.get("savefig") or kwargs.get("fname"):
        assert kwargs.get("fname")
        path = Path(kwargs["fname"])
        path.parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(str(kwargs["fname"]), code_img * 255)

    if show:
        plt.imshow(code_img, cmap="gray")
        plt.show()

    return code_img


def ax_imshow_with_colorbar(img, ax, fig, **imshow_kwargs):
    """
    Plotting colorbars in subplots
    :param img: Image to plot
    :param ax: mpl axis object
    :param fig: mpl figure object
    :param imshow_kwargs: passed to ax.imshow
    """
    im = ax.imshow(img, **imshow_kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


def visualize_point_cloud(
    points_3d,
    colors,
    view_kwargs: Dict = {},
    poisson_kwargs: Dict = {},
    savefig: bool = False,
    show: bool = True,
    create_mesh: bool = False,
    **kwargs,
):
    # 3D plot
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_3d)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
    pcd.remove_radius_outlier(nb_points=16, radius=0.1)

    if show:
        open3d.visualization.draw_geometries(
            [pcd], point_show_normal=True, **view_kwargs
        )

    if savefig:
        ply_path = Path(kwargs.get("fname") + "_point_cloud.ply")
        ply_path.parent.mkdir(exist_ok=True, parents=True)
        open3d.io.write_point_cloud(str(ply_path), pcd)

    if not create_mesh:
        return

    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd.estimate_normals()

    # Flip normals
    pcd.normals = open3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson meshing
    poisson_kwargs = {
        **dict(depth=9, width=0, scale=2, linear_fit=False),
        **poisson_kwargs,
    }
    (
        poisson_mesh,
        densities,
    ) = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, **poisson_kwargs
    )

    vertices_to_remove = densities < np.quantile(densities, 0.01)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    if show:
        open3d.visualization.draw_geometries([p_mesh_crop], **view_kwargs)

    if savefig:
        open3d.io.write_triangle_mesh(kwargs.get("fname") + "_mesh.ply", p_mesh_crop)


def plot_image_and_colorbar(
    img,
    fname,
    savefig: bool = False,
    show: bool = True,
    title: str = None,
    cbar_title: str = None,
    **imshow_kwargs,
):
    """
    Plot an image, with and without colorbar.
    Export colorbar too.

    :param img: H W C array
    :param fname: file name
    :param savefig: Whether to save the figure
    :param show: Display in graphical window or just close the plot
    :param title: optional plot title
    :param cbar_title: optional colorbar title
    :param imshow_kwargs:
    :return:
    """
    image = plt.imshow(img, **imshow_kwargs)
    plt.axis("off")

    outfolder = Path(fname).parent
    fname = Path(fname).stem
    save_plot(
        savefig,
        show=show,
        close=False,
        fname=f"{outfolder}/{fname}.pdf",
    )

    # Colorbar
    def img_colorbar(**kwargs):
        cbar = plt.colorbar(**kwargs)
        if cbar_title:
            cbar.ax.set_title(cbar_title)
        cbar.ax.locator_params(nbins=5)
        cbar.update_ticks()

    img_colorbar()
    if title:
        plt.title(title, y=1.12)
    save_plot(
        savefig,
        show,
        fname=f"{outfolder}/{fname}_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    img_colorbar(mappable=image, ax=ax)
    ax.remove()
    save_plot(
        savefig,
        show=False,
        fname=f"{outfolder}/{fname}_only_colorbar.pdf",
    )
