import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path

def convert(folder):
    """
    Reads dataset/human/batches_result_1.json and writes dataset/human/final_edges.json.
    Each visualized stroke (from 'final_proxies') becomes:
      "<id>": { "geometry": [[x,y,z],[x,y,z]], "id": <id> }
    """

    data_path = Path.cwd() / "dataset" / folder 
    file_path = data_path / "batches_result_1.json"

    with open(file_path, 'r') as f:
        all_data = json.load(f)

    final_edges = {}
    next_id = 0

    for data in all_data:
        for stroke in data.get("final_proxies", []):
            if isinstance(stroke, list) and len(stroke) == 2 \
               and all(isinstance(p, list) and len(p) == 3 for p in stroke):
                final_edges[str(next_id)] = {
                    "geometry": [stroke[0], stroke[1]],
                    "id": next_id
                }
                next_id += 1

    out_path = os.path.join(os.path.dirname(file_path), "final_edges.json")
    with open(out_path, "w") as f:
        json.dump(final_edges, f, indent=2)

    print(f"Wrote {len(final_edges)} edges to {out_path}")

def visualize_proxies(file_path="dataset/human/batches_result_1.json"):
    """
    Pure visualization. Plots 'final_proxies' in blue.
    """
    with open(file_path, 'r') as f:
        all_data = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_line(ax, seg, color='blue', style='solid', lw=1):
        ax.plot([seg[0][0], seg[1][0]],
                [seg[0][1], seg[1][1]],
                [seg[0][2], seg[1][2]],
                color=color, linestyle=style, linewidth=lw)

    for data in all_data:
        for stroke in data.get("final_proxies", []):
            if isinstance(stroke, list) and len(stroke) == 2 \
               and all(isinstance(p, list) and len(p) == 3 for p in stroke):
                plot_line(ax, stroke, 'blue')

    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    plt.tight_layout()
    plt.show()
