from io import BytesIO
from imageio import imread

import open3d as o3d
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame
import matplotlib
import seaborn as sns
import pyviz3d.visualizer as viz

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def point_cloud_plolty(
    coordinates, label_color, label_text, prediction_color, prediction_text, normals,
):
    def draw_point_cloud(coords, colors=None, label_text=None):
        marker = dict(size=1, opacity=0.8)
        if colors is not None:
            marker.update({"color": colors})
        if (colors is None) and (label_text is not None):
            marker.update({"color": label_text})
        fig = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            text=label_text,
            mode="markers",
            marker=marker,
        )
        return fig

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )
    fig.add_trace(
        draw_point_cloud(coordinates, prediction_color, prediction_text), row=1, col=1,
    )
    # adding image with prediction
    fig.add_trace(draw_point_cloud(coordinates, label_color, label_text), row=1, col=2)
    fig.show()
    # data = fig.to_image(width=1080, height=720, format="png")
    # image = Image.open(BytesIO(data))
    # return image


def point_cloud_pyviz3d(
    name,
    coordinates,
    path,
    color=None,
    normals=None,
    label_color=None,
    prediction_color=None,
    point_size=25,
    voxel_size=0.01,
):

    # because of visualization
    coordinates = coordinates * voxel_size
    # First, we set up a visualizer
    visualizer = viz.Visualizer()
    if label_color is not None:
        visualizer.add_points(
            name=f"{name}_label",
            positions=coordinates,
            colors=label_color,
            point_size=point_size,
            visible=False,
        )

    if prediction_color is not None:
        visualizer.add_points(
            name=f"{name}_prediction",
            positions=coordinates,
            colors=prediction_color,
            point_size=point_size,
            visible=False,
        )

    visualizer.add_points(
        name=name,
        positions=coordinates,
        colors=color,
        normals=normals,
        point_size=point_size,
        visible=False,
    )
    # When we added everything we need to the visualizer, we save it.
    visualizer.save(path, verbose=False)


def point_cloud_open3d(coordinates):
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coordinates))
    o3d.visualization.draw_geometries([points])


def _remap_model_output(output, labels):
    output = np.array(output)
    output_remapped = output.copy()
    for i, k in enumerate(labels.keys()):
        output_remapped[output == i] = k
    return output_remapped


def save_visualization(
    coordinates,
    name="none",
    color=None,
    normals=None,
    target=None,
    prediction=None,
    target_info=None,
    path="./saved",
    backend="pyviz3d",
    voxel_size=0.05,
    color_mean=[0.47793125906962, 0.4303257521323044, 0.3749598901421883],
    color_std=[0.2834475483823543, 0.27566157565723015, 0.27018971370874995],
):
    target = _remap_model_output(target, target_info)
    prediction = _remap_model_output(prediction, target_info)
    coordinates = coordinates[:, :3] - coordinates[:, :3].mean(axis=0)
    coordinates = coordinates * voxel_size
    if color is not None:
        color = (color * color_std + color_mean) * 255

    target_color = np.zeros((len(target), 3))
    target_text = np.full((len(target)), "empty")
    prediction_color = np.zeros((len(prediction), 3))
    prediction_text = np.full((len(prediction)), "empty")
    if target_info is not None:
        for k, v in target_info.items():
            target_color[target == k] = v["color"]
            target_text[target == k] = v["name"]
            prediction_color[prediction == k] = v["color"]
            prediction_text[prediction == k] = v["name"]
    if backend == "pyviz3d":
        point_cloud_pyviz3d(
            name=name,
            coordinates=coordinates,
            path=path,
            color=color,
            normals=normals,
            label_color=target_color,
            prediction_color=prediction_color,
            voxel_size=1,
        )
    elif backend == "plotly":
        point_cloud_plolty(
            coordinates=coordinates,
            normals=normals,
            label_color=target_color,
            label_text=target_text,
            prediction_color=prediction_color,
            prediction_text=prediction_text,
        )
    elif backend == "open3d":
        point_cloud_open3d(coordinates)
    else:
        print("No such backend")


def draw_confsion_matrix(confusion_matrix, label_db):
    index = [i for i in range(confusion_matrix.shape[0])]
    index = _remap_model_output(index, label_db)
    column_names = np.full((len(index)), "empty")
    for k, v in label_db.items():
        column_names[index == k] = v["name"]
    df_cm = DataFrame(confusion_matrix, index=column_names, columns=column_names)
    # pretty_plot_confusion_matrix(df_cm, fz=9)
    sns.heatmap(
        df_cm, annot=True, fmt="d", linewidths=0.25, annot_kws={"size": 5}, vmax=10000
    )
    buf = BytesIO()
    plt.savefig(buf, format="jpg")
    plt.close()
    buf.seek(0)
    image = imread(buf, format="jpg")
    buf.close()
    return image
