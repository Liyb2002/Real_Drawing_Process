from pathlib import Path
from build123d import *
import os
import numpy as np

from build123d import BuildSketch, BuildLine, Line, make_face



def build_sketch(count, canvas, Points_list, output, data_dir, tempt_idx=0):
    if tempt_idx == 0:
        brep_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")
    else:
        brep_dir = os.path.join(data_dir, "canvas", f"tempt_{tempt_idx}.step")

    if count == 0:
        with BuildSketch():
            with BuildLine():
                for i in range(0, len(Points_list), 2):
                    sp = Points_list[i]
                    ep = Points_list[i + 1]
                    Line((sp[0], sp[1], sp[2]), (ep[0], ep[1], ep[2]))

            perimeter = make_face()

        if output:
            export_step(perimeter, brep_dir)   # ✅ works in your version

        return perimeter

    else:
        with canvas:
            with BuildSketch():
                with BuildLine():
                    for i in range(0, len(Points_list), 2):
                        sp = Points_list[i]
                        ep = Points_list[i + 1]
                        Line((sp[0], sp[1], sp[2]), (ep[0], ep[1], ep[2]))

                perimeter = make_face()

            if output:
                export_step(perimeter, brep_dir)

        return perimeter





def build_circle(count, radius, point, normal, output, data_dir):
    if radius <= 0:
        raise ValueError("radius must be > 0")

    brep_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    # Build the circle on the specified plane
    with BuildSketch(
        Plane(origin=(point[0], point[1], point[2]),
              z_dir=(normal[0], normal[1], normal[2]))
    ) as s:
        Circle(radius=radius)

    # Make a planar face from the closed profile(s)
    face = s.face  # <- use the builder's face property

    if output:
        export_step(face, brep_dir)  # STEP surface of the circular face

    # Return the sketch object (if you need the face, return `face` instead)
    return s.sketch



def test_extrude(target_face, extrude_amount):

    with BuildPart() as test_canvas:
        extrude( target_face, amount=extrude_amount)

    return test_canvas

def build_extrude(count, canvas, target_face, extrude_amount, output, data_dir):
    stl_dir  = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    if canvas is not None:
        # Use existing BuildPart context
        with canvas:
            extrude(target_face, amount=extrude_amount)
    else:
        # Create a new BuildPart and extrude into it
        with BuildPart() as canvas:
            extrude(target_face, amount=extrude_amount)

    if output:
        # Use exporter FUNCTIONS (not Part methods)
        export_stl(canvas.part, stl_dir)
        export_step(canvas.part, step_dir)

    return canvas


def build_subtract(count, canvas, target_face, extrude_amount, output, data_dir):
    stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        extrude( target_face, amount= extrude_amount, mode=Mode.SUBTRACT)

    if output:
        _ = canvas.part.export_stl(stl_dir)
        _ = canvas.part.export_step(step_dir)


    return canvas

def build_fillet(count, canvas, target_edge, radius, output, data_dir):
    stl_dir  = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        fillet(target_edge, radius)

    if output:
        export_stl(canvas.part, stl_dir)
        export_step(canvas.part, step_dir)

    return canvas


def build_chamfer(count, canvas, target_edge, radius, output, data_dir):
    stl_dir  = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        chamfer(target_edge, radius)

    if output:
        export_stl(canvas.part, stl_dir)
        export_step(canvas.part, step_dir)

    return canvas


def simulate_extrude(sketch, amount):
    with BuildPart() as temp:
        extrude(sketch, amount=amount)
    return temp.part

def has_volume_overlap(canvas, new_part):
    intersection = canvas & new_part
    return intersection.volume > new_part.volume * 0.5
