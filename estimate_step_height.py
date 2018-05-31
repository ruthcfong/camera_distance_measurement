import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

CARD_CM = (8.560, 5.398) # (width, height)

TRANSFORMS = ['affine', 'perspective']


class Point(object):
    def __init__(self, p):
        super(Point, self).__init__()
        self.x = int(np.round(p[0]))
        self.y = int(np.round(p[1]))

    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __repr__(self):
        return '(%.2f, %.2f)' % (self.x, self.y)

    @staticmethod
    def get_slope(p1, p2):
        m = float(p2.y - p1.y) / (p2.x - p1.x)
        return m

    @staticmethod
    def get_distance(p1, p2):
        return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5

    @staticmethod
    def to_numpy(pt_arr):
        return np.float32([[pt.x, pt.y] for pt in pt_arr])

    @staticmethod
    def from_numpy(np_arr):
        return [Point(p) for p in np_arr]

    def to_tuple(self):
        return (self.x, self.y)


class Line(object):
    def __init__(self, p1=None, p2=None, m=None, b=None):
        super(Line, self).__init__()
        if m is None:
            assert(p1 is not None and p2 is not None)
            m = Point.get_slope(p1, p2)
        if b is None:
            b = -1*m*p1.x + p1.y

        self.m = m
        self.b = b

    def __repr__(self):
        return "y = %.4f*x + %.4f" % (self.m, self.b)

    def get_y(self, x):
        return int(np.round(self.m * x + self.b))

    def intersect(self, other):
        x = float(self.b - other.b) / (other.b - self.b)
        y = self.get_y(x)
        print y, other.get_y(x)
        return Point((int(np.round(x)),int(np.round(y))))
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, metavar='IMG_PATH',
                        help='path to image')
    parser.add_argument('--transform', type=str, metavar='TRANSFORM', default='perspective',
                        choices=TRANSFORMS,
                        help='type of transform')

    opts = parser.parse_args()

    return opts

def get_clicks(img_path, num_clicks=None, text=None, print_pos=False):
    
    if text is not None:
        print text

    print('Press any key when finished clicking (press "r" to restart clicking).') 
    img = cv2.imread(img_path)

    clicks = []
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if print_pos:
                print('(%d, %d)' % (x, y))
            clicks.append((x,y))

            # draw small blue circle at mouse position when clicked
            cv2.circle(img, (x,y), 5, (255, 0, 0), -1) 
            cv2.imshow('image', img)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while(True):
        cv2.imshow('image', img)
        key = cv2.waitKey(0)

        if key == ord("r"):
            clicks = []
            img = cv2.imread(img_path)
            cv2.imshow('image', img)
        elif num_clicks is not None and len(clicks) < num_clicks:
            print('Keep clicking to get to %d clicks.' % num_clicks)
        else:
            if num_clicks is not None and len(clicks) > num_clicks:
                print('Using the first %d clicks.' % num_clicks)
                clicks = clicks[:num_clicks]
            break

    cv2.destroyAllWindows()

    return clicks


def transform(opts):
    # Get clicks from user
    instructions='\nClick the following 8 points in order:\n'\
            'A. Credit card (1., top left corner, 2., top right corner, 3. bottom right corner, 4., bottom left corner)\n'\
            'B. Bottom step (5., a leftmost point, 6., a rightmost point)\n'\
            'C. Top step (7., a leftmost point, 8., a rightmost point)\n'
        
    clicks = get_clicks(opts.image_path, num_clicks=8, text=instructions, print_pos=False)

    card_orig = Point.from_numpy(clicks[:4])
    steps_orig = Point.from_numpy(clicks[4:])

    tl, tr, bl, br = card_orig
    
    card_width_pixels_orig = int(np.round(Point.get_distance(tl, tr)))
    card_height_pixels_orig = int(np.round(Point.get_distance(tl, bl)))

    card_deltas = Point.from_numpy([(0,0), 
                                    (card_width_pixels_orig, 0), 
                                    (0, card_height_pixels_orig), 
                                    (card_width_pixels_orig, card_height_pixels_orig)])

    card_new = [tl + d for d in card_deltas]

    if opts.transform == 'affine':
        M = cv2.getAffineTransform(Point.to_numpy(card_orig[:3]), Point.to_numpy(card_new[:3]))
    elif opts.transform == 'perspective':
        M = cv2.getPerspectiveTransform(Point.to_numpy(card_orig), Point.to_numpy(card_new))

    img = cv2.imread(opts.image_path)
    height, width, ch = img.shape
    if opts.transform == 'affine':
        new_img = cv2.warpAffine(img, M,  (width, height))
    elif opts.transform == 'perspective':
        new_img = cv2.warpPerspective(img, M,  (width, height))

    new_height, new_width, new_ch = new_img.shape

    if opts.transform == 'affine':
        M_pers = np.identity(3)
        M_pers[:2,:3] = M
        M = M_pers

    steps_new = Point.from_numpy(cv2.perspectiveTransform(np.array([Point.to_numpy(steps_orig)]), M)[0])

    w, x, y, z = steps_new
    m1 = Point.get_slope(x, w)
    m2 = Point.get_slope(y, z)

    bottom_step = Line(p1=w, m=m1)
    top_step = Line(p1=y, m=m2)

    bottom_perp_point = Point((card_new[2].x, bottom_step.get_y(card_new[2].x)))
    top_perp_point  = Point((card_new[2].x, top_step.get_y(card_new[2].x)))

    step_dist_pixels = Point.get_distance(bottom_perp_point, top_perp_point)
    card_height_pixels = Point.get_distance(card_new[0], card_new[2]) 
    step_dist_cm = CARD_CM[1] / card_height_pixels * step_dist_pixels

    print('Step height: %.2f cm' % step_dist_cm) 

    # Draw on warped image 
    cv2.rectangle(new_img, card_new[0].to_tuple(), card_new[-1].to_tuple(), (0, 255, 0), 3)
    cv2.line(new_img, top_perp_point.to_tuple(), bottom_perp_point.to_tuple(), (0, 0, 255), 3)
    cv2.line(new_img, (0, top_step.get_y(0)), (new_width, top_step.get_y(new_width)), (0, 0, 255), 3)
    cv2.line(new_img, (0, bottom_step.get_y(0)), (new_width, bottom_step.get_y(new_width)), (0, 0, 255), 3)
    cv2.circle(new_img, top_perp_point.to_tuple(), 5, (255, 0, 0), -1) 
    cv2.circle(new_img, bottom_perp_point.to_tuple(), 5, (255, 0, 0), -1) 
    cv2.imshow('image', new_img)

    print('Press any key to exit.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opts = parse_args()
    transform(opts)

