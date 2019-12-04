import uuid
import os
import math


def create_tmp_file(dir, suffix=None):
    filename = str(uuid.uuid4())
    if suffix is not None:
        filename += '.{}'.format(suffix)
    return os.path.join(dir, filename)


# center - x0,y0
def get_angle_in_circle(x, y, x0, y0):
    (dx, dy) = (x0-x, y-y0)
    # angle = atan(float(dy)/float(dx))
    angle = math.degrees(math.atan2(float(dy), float(dx)))
#     if angle < 0:
#         angle += 180
    return angle
