import numpy as np
import math

from pathlib import Path


def generate_xodr_from_segments(segments, track_width=1.1, slack=0.15):
    """
    Generate an OpenDRIVE (.xodr) file from a list of segments.

    Args:
        segments (list of [length, radius]): Nx2 matrix describing the track segments.
        track_width (float): Width of the track lanes.

    Returns:
        str: The .xodr file content as a string.
    """
    x, y, heading = 0, 0, 0  # Initial position and heading
    cum_s = 0  # Cumulative s-coordinate
    road_geometry = []

    for length, radius in segments:
        if radius == 0:  # Straight segment
            dx = length * math.cos(heading)
            dy = length * math.sin(heading)
            x_end = x + dx
            y_end = y + dy

            road_geometry.append(f"""
            <geometry s="{cum_s}" x="{x}" y="{y}" hdg="{heading}" length="{length}">
              <line/>
            </geometry>""")

            x, y = x_end, y_end
            cum_s += length

        else:  # Curved segment
            curvature = 1 / radius
            # curvature = radius
            arc_angle = length / radius
            cx = x - radius * math.sin(heading)  # Center of the arc
            cy = y + radius * math.cos(heading)

            # Compute end position
            x_end = cx + radius * math.sin(heading + arc_angle)
            y_end = cy - radius * math.cos(heading + arc_angle)

            road_geometry.append(f"""
            <geometry s="{cum_s}" x="{x}" y="{y}" hdg="{heading}" length="{length}">
              <arc curvature="{curvature}"/>
            </geometry>""")

            x, y = x_end, y_end
            heading += arc_angle
            cum_s += length

    total_length = cum_s
    xodr_content = f"""<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="SegmentTrack" version="1.00" date="2025-01-20" north="0" south="0" east="0" west="0"/>
  <road name="SegmentTrack" length="{total_length}" id="1" junction="-1">
    <planView>
      {''.join(road_geometry)}
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="{track_width/2 + slack:.2f}" b="0.0" c="0.0" d="0.0"/>
          </lane>
        </right>
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="{track_width/2 + slack:.2f}" b="0.0" c="0.0" d="0.0"/>
          </lane>
        </left>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>"""
    return xodr_content


def generate_LTrack_barc_xodr():
    # Track parameters
    track_width = 1.1
    slack     = 0.15 # 0.3

    ninety_radius_1     = ((1.5912+0.44723)/2 + (1.5772+0.46504)/2)/2 
    ninety_radius_2     = ((0.65556 + 1.12113/2) + (0.6597 + 1.13086/2))/2

    oneeighty_radius_1  = (1.171 + 1.1473/2 + 1.1207/2)/2
    oneeighty_radius_2  = (1.3165 + 1.15471/2 + 1.12502/2)/2

    straight_1          = 2.401 - 0.15
    straight_2          = 1.051 - 0.15
    straight_3          = 0.450 - 0.3
    straight_4          = 2*oneeighty_radius_1 + ninety_radius_1 + straight_3 - ninety_radius_2 #2.5515
    straight_5          = np.abs(straight_1 - straight_2 - ninety_radius_1 - 2*oneeighty_radius_2 + ninety_radius_2)

    cl_segs = np.array([[straight_1,                    0],
                        [np.pi*oneeighty_radius_1,      oneeighty_radius_1],
                        [straight_2,                    0],
                        [np.pi/2*ninety_radius_1,       -ninety_radius_1],
                        [straight_3,                    0],
                        [np.pi*oneeighty_radius_2,      oneeighty_radius_2],
                        [straight_4,                    0],
                        [np.pi/2*ninety_radius_2,       ninety_radius_2],
                        [straight_5,                    0]])

    # Generate and save the .xodr file
    xodr_content = generate_xodr_from_segments(cl_segs, track_width=track_width, slack=slack)
    with open("L_track_barc.xodr", "w") as file:
        file.write(xodr_content)

    print("XODR file created: L_track_barc.xodr")


def generate_xodr_from_arc_length_track_data(track_name):
    from mpclab_common.track import get_track
    track_obj = get_track(track_name)
    if track_obj.__class__.__name__ != 'RadiusArclengthTrack':
        raise TypeError("This function only works with RadiusArclengthTrack!")
    cl_segs = track_obj.cl_segs
    track_width = track_obj.track_width.item()
    # slack = track_obj.slack.item()
    xodr_content = generate_xodr_from_segments(cl_segs, track_width=track_width, slack=0.15)
    target_path = Path(__file__).resolve().parents[1] / "OpenDrive" / f"{track_name}.xodr"
    print(target_path.resolve())
    with open(target_path, "w") as f:
        f.write(xodr_content)
    print(f"XODR file created: {track_name}.xodr")


if __name__ == '__main__':
    # generate_LTrack_barc_xodr()
    generate_xodr_from_arc_length_track_data("L_track_barc")
