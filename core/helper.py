def intersect(A, B, C, D):
    """
    Check if line segment AB intersects with CD
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A, B, C):
    """
    Check clockwise orientation of an ordered triplet (A, B, C)
    """
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def detect_class(center, classes, confidences, mid_points):
    for i in range(len(classes)):
        if abs(center[0] - mid_points[i][0]) < 7 and abs(center[1] - mid_points[i][1]) < 7:
            return classes[i], confidences[i]
    
    return "", 0