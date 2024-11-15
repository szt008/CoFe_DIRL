import numpy as np

def signedDistToLine(point, line_point, line_dir_vec) -> float:
    """Computes the signed distance to a directed line
    The signed of the distance is:

      - negative if point is on the right of the line
      - positive if point is on the left of the line

    ..code-block:: python

        >>> import numpy as np
        >>> signed_dist_to_line(np.array([2, 0]), np.array([0, 0]), np.array([0, 1.]))
        -2.0
        >>> signed_dist_to_line(np.array([-1.5, 0]), np.array([0, 0]), np.array([0, 1.]))
        1.5
    """
    p = vec_2d(point)
    p1 = line_point
    p2 = line_point + line_dir_vec

    u = abs(
        line_dir_vec[1] * p[0] - line_dir_vec[0] * p[1] + p2[0] * p1[1] - p2[1] * p1[0]
    )
    d = u / np.linalg.norm(line_dir_vec)

    line_normal = np.array([-line_dir_vec[1], line_dir_vec[0]])
    _sign = np.sign(np.dot(p - p1, line_normal))
    return d * _sign


def vec_2d(v) -> np.ndarray:
    """Converts a higher order vector to a 2D vector."""

    assert len(v) >= 2
    return np.array(v[:2])

def linearInterpolation(a, b, rate_of_a):
    return a * rate_of_a + b * (1 - rate_of_a)