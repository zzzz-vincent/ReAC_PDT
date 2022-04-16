

import os
import numpy as np
import open3d as o3d
from mayavi import mlab


def boxes_to_corners(boxes, resize_factor=1.0, connect_inds=False):
    c_xyz = np.array(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )  # (3, 8)
    c_xyz = 0.5 * c_xyz[np.newaxis, :, :] * boxes[:, 3:6, np.newaxis]  # (B, 3, 8)
    c_xyz = c_xyz * resize_factor

    # to world frame
    R = get_R(boxes)  # (B, 3, 3)
    c_xyz = R @ c_xyz + boxes[:, :3, np.newaxis]  # (B, 3, 8)

    if not connect_inds:
        return c_xyz
    else:
        l1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 1]
        l2 = [0, 5, 1]
        l3 = [2, 6]
        l4 = [3, 7]
        return c_xyz, (l1, l2, l3, l4)


def get_R(boxes):
    theta = boxes[:, 6] + np.pi
    cs, ss = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros(len(cs)), np.ones(len(cs))
    Rs = np.array(
        [[cs, ss, zeros], [-ss, cs, zeros], [zeros, zeros, ones]], dtype=np.float32
    )  # (3, 3, B)

    return Rs.transpose((2, 0, 1))


def string_to_boxes(s, jrdb_format=True, get_num_points=False):
    boxes = []
    scores = []

    lines = s.split("\n")
    for line in lines:
        if len(line) == 0:
            continue
        v_list = [float(v) for v in line.split()[-8:]]
        scores.append(v_list[-1])
        boxes.append([v_list[i] for i in [3, 4, 5, 0, 1, 2, 6]])

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    if jrdb_format and len(boxes) > 0:
        boxes = convert_boxes_kitti_to_jrdb(boxes)

    if get_num_points:
        num_points = np.array([int(line.split()[3]) for line in lines if len(line) > 0])
        return boxes, scores, num_points
    else:
        return boxes, scores

def convert_boxes_kitti_to_jrdb(kitti_boxes):
    jrdb_boxes = np.stack(
        [
            kitti_boxes[:, 2],
            -kitti_boxes[:, 0],
            -kitti_boxes[:, 1] + 0.5 * kitti_boxes[:, 3],
            kitti_boxes[:, 5],
            kitti_boxes[:, 4],
            kitti_boxes[:, 3],
            -kitti_boxes[:, 6],
        ],
        axis=1,
    )

    return jrdb_boxes

############################################ start here ############################################

if __name__ == "__main__":


    gs_blue = (66.0 / 256, 133.0 / 256, 244.0 / 256)
    gs_gray = (220 / 256, 220 / 256, 220 / 256)
    gs_red = (234.0 / 256, 68.0 / 256, 52.0 / 256)
    gs_yellow = (251.0 / 256, 188.0 / 256, 4.0 / 256)
    gs_green = (52.0 / 256, 168.0 / 256, 83.0 / 256)
    gs_orange = (255.0 / 256, 109.0 / 256, 1.0 / 256)
    gs_blue_light = (70.0 / 256, 189.0 / 256, 196.0 / 256)

    fig = mlab.figure(
        figure=None,
        bgcolor=(1, 1, 1),
        fgcolor=(0, 0, 0),
        engine=None,
        size=(800, 900),
    )

    root_path = '/Users/zishenwen/Work/CMU/ReAC/Person_MinkUNet-main/test_result'
    upper_path = '/Users/zishenwen/Work/CMU/ReAC/Person_MinkUNet-main/pointcloud/lower/cubberly-auditorium-2019-04-22_1'
    lower_path = '/Users/zishenwen/Work/CMU/ReAC/Person_MinkUNet-main/pointcloud/lower/cubberly-auditorium-2019-04-22_1'
    # root_path = '/Users/jackiexu/PycharmProjects/test_result'

    i = 0
    frames = []
    for filename in os.listdir(root_path): # open folder
        file_path = os.path.join(root_path, filename)
        if not filename.startswith('.') and os.path.isfile(file_path):
            frame = filename.split(".")[0]
            frames.append(frame)
    frames.sort()

    print("total frames:", len(frames))

    for frame in frames:
        print("processing frame:", frame)
        upper_file = upper_path + "/" + frame + ".pcd"
        lower_file = lower_path + "/" + frame + ".pcd"
        box_file = root_path + "/" + frame + ".txt"
        pc_lower = np.asarray(o3d.io.read_point_cloud(lower_file).points)
        pc_upper = np.asarray(o3d.io.read_point_cloud(upper_file).points)
        pts = np.concatenate([pc_upper, pc_lower], axis=0).transpose()  # (3, N)

        s = np.hypot(pts[0], pts[1])
        mpt = mlab.points3d(
            pts[0],
            pts[1],
            pts[2],
            s,
            colormap="blue-red",
            mode="sphere",
            scale_factor=0.06,
            figure=fig,
        )
        print("rendered points...")

        with open(box_file, 'r') as f: # open file
            boxes_all, scores_all = string_to_boxes(f.read())
            corners_xyz, connect_inds = boxes_to_corners(boxes_all, connect_inds=True)

            for corner_xyz, score in zip(corners_xyz, scores_all):
                for inds in connect_inds:
                    plt = mlab.plot3d(
                        corner_xyz[0, inds],
                        corner_xyz[1, inds],
                        corner_xyz[2, inds],
                        tube_radius=None,
                        line_width=3,
                        color=gs_yellow if score > 0.1 else gs_gray,
                        figure=fig,
                    )
        print("rendered boxes...")

        mlab.view(
            azimuth=180,
            elevation=70,
            focalpoint=[12.0909996, -1.04700089, -2.03249991],
            distance=62,
            figure=fig,
        )

        # mlab.show()
        mlab.savefig(frame + '.png')
        print("saved", frame)



