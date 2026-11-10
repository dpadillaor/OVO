"""ovo.eval.fusion.projection: map points to one mask per instance on the mesh."""

import numpy as np

from ovo.eval.fusion.projection import project


def test_masks_follow_the_nearest_points():
    line = lambda x0, n: [[x0 + 0.01 * k, 0.0, 0.0] for k in range(n)]
    points = np.array(line(0, 6) + line(1, 6) + line(5, 1) + line(20, 1), dtype=float)
    ids = np.array([3] * 6 + [7] * 6 + [-1] + [8])  # -1 unassigned; 8 is far from every vertex
    vertices = np.array([[0.02, 0, 0], [1.02, 0, 0], [0.03, 0, 0], [5.0, 0, 0]])

    masks, out_ids = project(points, ids, vertices)

    assert out_ids.tolist() == [3, 7]  # 8 reaches no vertex: no mask
    assert masks.dtype == bool
    # the vertex at x=5 sits on the unassigned point, which does not vote: 7 is nearer than 3
    assert masks.tolist() == [[True, False, True, False], [False, True, False, True]]
