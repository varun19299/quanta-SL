import numpy as np
import plotly.graph_objects as go
from quanta_SL.utils.decorators import named_func
from pathlib import Path


@named_func("P(y = 0 | x = 1)", "Bit-flip Probability")
def p_y_0_x_1(phi_A, phi_P, t_exp: float = 1e-4, eta: float = 1.0, r_q: float = 0.0):
    return np.exp(-(eta * phi_P + r_q) * t_exp)


@named_func("P(y = 1 | x = 0)", "Bit-flip Probability")
def p_y_1_x_0(phi_A, phi_P, t_exp: float = 1e-4, eta: float = 1.0, r_q: float = 0.0):
    return 1 - np.exp(-(eta * (phi_A + r_q) * t_exp))


def surface_plot(
    error_func,
    phi_A,
    phi_proj,
    title: str = "",
    color: str = "blue",
    show: bool = True,
    savefig: bool = True,
    **eval_func_kwargs,
):
    print(1)
    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
    phi_proj_mesh, phi_A_mesh = np.meshgrid(phi_proj, phi_A, indexing="ij")

    color_discrete = [(0, color), (1, color)]

    eval_error = error_func(phi_A_mesh, phi_P_mesh, **eval_func_kwargs)

    fig = go.Figure(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            autosize=False,
            width=900,
            height=900,
        ),
    )

    line_marker = dict(color="black", width=2)
    for i, j, k in zip(phi_proj_mesh[::4], phi_A_mesh[::4], eval_error[::4]):
        fig.add_trace(
            go.Scatter3d(
                x=i, y=j, z=k, mode="lines", line=line_marker, showlegend=False
            )
        )

    for i, j, k in zip(phi_proj_mesh.T[::4], phi_A_mesh.T[::4], eval_error.T[::4]):
        fig.add_trace(
            go.Scatter3d(
                x=i, y=j, z=k, mode="lines", line=line_marker, showlegend=False
            )
        )

    fig.add_trace(
        go.Surface(
            x=phi_proj_mesh,
            y=phi_A_mesh,
            z=np.round(eval_error, decimals=3),
            name=error_func.name,
            opacity=1,
            colorscale=color_discrete,
            showlegend=True,
            showscale=False,
        )
    )

    fig.update_layout(
        # showlegend=True,
        title=dict(text=title, x=0.5, y=0.9, xanchor="center", yanchor="top"),
        scene=dict(
            xaxis=dict(
                title=r"Projector Flux",
                tickfont_size=12,
                dtick="D10",
                type="log",
                exponentformat="power",
            ),
            yaxis=dict(
                title=r"Ambient Flux",
                tickfont_size=12,
                dtick="D10",
                type="log",
                exponentformat="power",
            ),
            zaxis=dict(
                title=error_func.long_name,
                tickfont_size=12,
                # type="log",
            ),
        ),
        scene_aspectmode="cube",
        scene_camera_eye=dict(x=1.61, y=1.61, z=0.25),
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.87),
        font=dict(size=18),
    )

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    # fig.update_traces(showlegend=True)

    if show:
        fig.show(renderer="browser")

    if savefig:
        plot_dir = Path("outputs/plots")
        out_path = plot_dir / f"{error_func.name}"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_image(str(out_path) + ".pdf", scale=4)
        fig.write_image(str(out_path) + ".png", scale=4)
        fig.write_html(str(out_path) + ".html", include_plotlyjs="cdn")


if __name__ == "__main__":
    COLORS_ll = ["red", "orange", "green", "blue", "purple", "brown", "grey"]

    num = 128
    phi_proj = np.logspace(1, 5, num=num)
    phi_A = np.logspace(1, 5, num=num)

    phi_proj = np.logspace(1.5, 5.5, num=num)
    phi_A = np.logspace(1.5, 5.5, num=num)

    surface_plot(p_y_0_x_1, phi_A, phi_proj, color="orange")
    surface_plot(p_y_1_x_0, phi_A, phi_proj, color="purple")
