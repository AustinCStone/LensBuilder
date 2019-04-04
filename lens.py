import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict 
import sys
import pickle

sys.setrecursionlimit(100000)

SLOWDOWN_COEFF = .25

NUM_LENS_PTS = 20

# world point
WP = (50., 1000.)

# starting angle of incidence of vertical rays with lens
# indexed by segment end point
SEGMENT_LEN = 1. / NUM_LENS_PTS
LENS_ANGLES = OrderedDict()
angles = np.linspace(.5, -.5, NUM_LENS_PTS)
for i, x_pos in enumerate(np.linspace(SEGMENT_LEN, 1., NUM_LENS_PTS)):
    LENS_ANGLES[x_pos] = angles[i]

print(LENS_ANGLES)
# focal point
FOCAL = (0.5, -5.)


def get_lens_angle(query_x_pos):
    x_end_pos = LENS_ANGLES.keys()
    x_start_pos = np.asarray(x_end_pos) - 1. / NUM_LENS_PTS
    for i, (x_start, x_end) in enumerate(zip(x_start_pos, x_end_pos)):
        if x_start <= query_x_pos < x_end:
            if i == 0 and query_x_pos < SEGMENT_LEN / 2. + 1e-4:
                return LENS_ANGLES[x_end]
            if i == NUM_LENS_PTS - 1 and x_end - query_x_pos <= SEGMENT_LEN / 2. + 1e-4:
                return LENS_ANGLES[x_end]

            # interpolate current segment and next segment
            if x_end - query_x_pos > SEGMENT_LEN / 2.:
                first_weight = 1. - (query_x_pos - x_start) / (SEGMENT_LEN / 2.)
                second_weight = 1. - first_weight
                first_key = LENS_ANGLES.keys()[i - 1]
                return first_weight * LENS_ANGLES[first_key] + second_weight * LENS_ANGLES[x_end]
            else:
                first_weight = (x_end - query_x_pos) / (SEGMENT_LEN / 2.)
                second_weight = 1. - first_weight
                second_key = LENS_ANGLES.keys()[i + 1]
                return first_weight * LENS_ANGLES[x_end] + second_weight * LENS_ANGLES[second_key]
    raise ValueError


def get_angle_refraction(ray_angle, lens_angle, coeff):
    # rotate lens to be horizontal
    # ray angle is positive when rotated to the left, lens angle is positive
    # when left side is facing down
    ray_angle += lens_angle
    sin_refraction = np.sin(ray_angle) * SLOWDOWN_COEFF
    angle_refraction = np.arcsin(sin_refraction)
    return angle_refraction - lens_angle


def get_lens_height(query_x_pos):
    height = 0.
    segment_x_len = 1. / NUM_LENS_PTS
    for x_end_pos, angle in LENS_ANGLES.items():
        start_height = height
        end_height = np.tan(angle) * segment_x_len
        segment_end = (x_end_pos, start_height + end_height)
        height = segment_end[1]
        query_in_front = query_x_pos >= x_end_pos - segment_x_len
        query_behind = segment_end[0] >= query_x_pos
        if query_in_front and query_behind:
            interp_x = (segment_end[0] - query_x_pos) / segment_x_len
            return interp_x * start_height + (1. - interp_x) * segment_end[1]
    import ipdb
    ipdb.set_trace()
    raise ValueError


def binary_search_exit_pt(ray_start, ray_angle, lo_exit_y, hi_exit_y,
                          tolerance=1e-4, x_min=0., x_max=1.):
    mid_y = (lo_exit_y + hi_exit_y) / 2.
    mid_x = np.tan(ray_angle) * mid_y + ray_start[0]
    mid_x = np.clip(mid_x, 0., 1.)
    height = -get_lens_height(mid_x)
    if abs(height - mid_y) < tolerance:
        return (mid_x, mid_y)
    if height < mid_y:
        diff = height - mid_y
        lo_exit_y = mid_y
    else: 
        hi_exit_y = mid_y
    return binary_search_exit_pt(ray_start, ray_angle, lo_exit_y, hi_exit_y)


def get_ray_angle(ray):
    ray_start, ray_end = ray
    ray_x = ray_end[0] - ray_start[0]
    ray_y = ray_end[1] - ray_start[1]
    ray_angle = np.arctan(ray_x / ray_y)
    return ray_angle


def get_interior_rays(ray_start_end):
    """Rays inside the lens"""
    interior_rays = []
    for i, (ray_start, ray_end) in enumerate(ray_start_end):
        entry_point = ray_end
        ray_angle = get_ray_angle((ray_start, ray_end))
        angle_refraction = get_angle_refraction(ray_angle, get_lens_angle(ray_end[0]), SLOWDOWN_COEFF)
        exit_point = binary_search_exit_pt(entry_point, angle_refraction, lo_exit_y=ray_start[1],
                                           hi_exit_y=FOCAL[1], x_min=0., x_max=1.)
        interior_rays.append((entry_point, exit_point))
    return interior_rays


def get_focusing_rays(interior_rays):
    """Takes in list of start, end positions of rays inside the lens.
    Returns start, end positions of rays leaving the lens where ray's
    end position is capped at the focal length."""
    focusing_rays = []
    for i, (ray_start, ray_end) in enumerate(interior_rays):
        ray_angle = get_ray_angle((ray_start, ray_end))
        exterior_angle_of_refraction = get_angle_refraction(ray_angle, -get_lens_angle(ray_end[0]), 1. / SLOWDOWN_COEFF)
        #ray_end_y = ray_end[1] - np.cos(exterior_angle_of_refraction)
        #ray_end_x = ray_end[0] + np.sin(exterior_angle_of_refraction)
        multiply_factor = abs(FOCAL[1] - ray_end[1]) / np.cos(exterior_angle_of_refraction)
        ray_end_y = ray_end[1] - multiply_factor * np.cos(exterior_angle_of_refraction)
        ray_end_x = ray_end[0] + multiply_factor * np.sin(exterior_angle_of_refraction)
        #ray_end_y *= multiply_factor
        #ray_end_x *= multiply_factor
        focusing_rays.append((ray_end, (ray_end_x, ray_end_y)))
    return focusing_rays


def get_rays():
    """Returns a list of rays moving from WP to the lens at equal spacings."""
    segment_x_len = 1. / NUM_LENS_PTS
    ray_start_end = []
    for x_end_pos in LENS_ANGLES.keys():
        ray_hit_x = x_end_pos - segment_x_len / 2.
        lens_height = get_lens_height(ray_hit_x)
        ray_start = WP
        ray_end = (ray_hit_x, lens_height)
        ray_start_end.append((ray_start, ray_end))
    return ray_start_end


def draw_lens(show_rays=True):
    lens_height = 0.
    num_pts = len(LENS_ANGLES)
    segment_x_len = 1. / NUM_LENS_PTS
    x_end_positions = LENS_ANGLES.keys()
    x_start_positions = np.asarray(x_end_positions) - segment_x_len
    for x_start, x_end in zip(x_start_positions, x_end_positions):
        height1 = get_lens_height(x_start)
        height2 = get_lens_height(x_end)
        plt.plot([x_start, x_end], [height1, height2])
        plt.plot([x_start, x_end], [-height1, -height2])

    plt.scatter([WP[0]], [WP[1]])
    plt.scatter([FOCAL[0]], [FOCAL[1]])

    if show_rays:
        rays = get_rays()
        interior_rays = get_interior_rays(rays)
        focusing_rays = get_focusing_rays(interior_rays)
        for ray_start, ray_end in rays:
            plt.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]])
        for ray_start, ray_end in interior_rays:
            plt.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]])
        for ray_start, ray_end in focusing_rays:
            plt.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]])
    plt.show()


def binary_search_angles(min_angles=None, max_angles=None, tolerance=.004):
    global LENS_ANGLES
    if min_angles is None:
        min_angles = []
        for i, k in enumerate(LENS_ANGLES.keys()):
            if i < NUM_LENS_PTS // 2:
                min_angles.append((k, 0.))
            else:
                min_angles.append((k, -np.pi / 2.5))
        min_angles = OrderedDict(min_angles)
    if max_angles is None:
        max_angles = []
        for i, k in enumerate(LENS_ANGLES.keys()):
            if i < NUM_LENS_PTS // 2:
                max_angles.append((k, np.pi / 2.5))
            else:
                max_angles.append((k, 0.))
        max_angles = OrderedDict(max_angles)

    LENS_ANGLES = OrderedDict([(k, (min_angles[k] + max_angles[k]) / 2.) for k in LENS_ANGLES.keys()])
    rays = get_rays()
    interior_rays = get_interior_rays(rays)
    focusing_rays = get_focusing_rays(interior_rays)
    done = True
    for i, (start, end) in enumerate(focusing_rays):
        print(abs(end[0] - FOCAL[0]))
        if abs(end[0] - FOCAL[0]) < tolerance: # good angle
            continue
        done = False
        key = LENS_ANGLES.keys()[i]
        # TODO: seems like newtons method would work here?
        if end[0] > FOCAL[0]: # angle needs to be more positive
            min_angles[key] += .0005
        else: # angle needs to be more negative
            max_angles[key] -= .0005
    if done:
        return
    binary_search_angles(min_angles, max_angles)


def pickle_lens():
    with open('lens10.pickle', 'wb') as handle:
        pickle.dump(LENS_ANGLES, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle():
    global LENS_ANGLES
    with open('lens10.pickle', 'rb') as handle:
        LENS_ANGLES = pickle.load(handle)

def main():
    unpickle()
    #binary_search_angles()
    print(LENS_ANGLES)
    draw_lens()
    #pickle_lens()

if __name__ == '__main__':
    main()


    

