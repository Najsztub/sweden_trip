def line(p1, p2):
    # Generate line equation Ax + By = C from two given points
    return {
        'a': p2[1] - p1[1],
        'b': p1[0] - p2[0],
        'c': p1[0] * p2[1] - p1[1] * p2[0]
    }

def line_intersection(l1, l2):
    # Return line intersection coordinates,
    # None in case lines do not intersect
    # I use Cramer's rule
    det = l1['a'] * l2['b'] - l1['b'] * l2['a']
    if abs(det) <  1e-10:
        # Lines parallel
        return None
    x0 = (l1['c'] * l2['b'] - l1['b'] * l2['c']) / det
    y0 = (l1['a'] * l2['c'] - l1['c'] * l2['a']) / det
    return (x0, y0)