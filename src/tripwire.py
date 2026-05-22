"""
Tripwire line definition and crossing detection.
"""
import cv2


class Tripwire:
    """
    A virtual line segment between two points. Detects when a tracked point
    crosses from one side to the other.
    
    Side detection uses the 2D cross product:
        For line A->B and point P:
            cross = (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x)
        cross > 0 -> P is on one side
        cross < 0 -> P is on the other side
        cross == 0 -> P is exactly on the line
    
    When the sign of `cross` flips between consecutive frames for the same
    tracked object, a crossing event has occurred.
    """
    
    def __init__(self, point_a, point_b, name="tripwire"):
        self.a = point_a  # (x, y)
        self.b = point_b
        self.name = name
    
    def side(self, point):
        """Return +1, -1, or 0 depending on which side of the line `point` is on."""
        ax, ay = self.a
        bx, by = self.b
        px, py = point
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0
    
    def crossed(self, prev_point, curr_point):
        """
        Returns a crossing direction if the object crossed the line, else None.
        Direction: +1 = positive-side to negative-side, -1 = the opposite.
        """
        if prev_point is None or curr_point is None:
            return None
        s1 = self.side(prev_point)
        s2 = self.side(curr_point)
        if s1 != 0 and s2 != 0 and s1 != s2:
            return s1  # the side it came FROM
        return None
    
    def draw(self, frame, color=(0, 255, 255), thickness=2):
        cv2.line(frame, self.a, self.b, color, thickness)
        cv2.putText(frame, self.name, (self.a[0] + 5, self.a[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame


def define_tripwire_interactively(frame, window_name="Click two points to define tripwire"):
    """
    Show a frame and let the user click two points to define a tripwire.
    Returns ((x1, y1), (x2, y2)).
    """
    points = []
    display = frame.copy()
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 255), 2)
            cv2.imshow(window_name, display)
    
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)
    
    print("Click two points on the image to define the tripwire line.")
    print("Press any key after both points are placed.")
    while len(points) < 2:
        cv2.waitKey(20)
    cv2.waitKey(0)  # wait for any key after both points
    cv2.destroyWindow(window_name)
    return points[0], points[1]