import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pygame

TRANSFORMS = ['affine', 'perspective']

'''
def get_slope(p1, p2):
    m = float(p2.y - p1.y) / (p2.x - p1.x)
    return m


def get_distance(p1, p2):
    return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5


class Point(object):
    def __init__(p):
        super(Point, self).__init__()
        self.x = p[0]
        self.y = p[1]


class Line(object):
    def __init__(p1=None, p2=None, m=None, b=None):
        super(Line, self).__init__()
        assert((p1 is not None and p2 is not None) or (m is not None and b is not None)) 
        if p1 is not None and p2 is not None:
            m = get_slope(p1, p2)
            b = -1*m*p1.x + p1.y
        self.m = m
        self.b = b
'''
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, metavar='IMG_PATH',
                        help='path to image')
    parser.add_argument('--transform', type=str, metavar='TRANSFORM', default='perspective',
                        choices=TRANSFORMS,
                        help='type of transform')

    opts = parser.parse_args()

    return opts
def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px


def loop_clicks(screen, px, num_clicks=4):
    pos = []
    while len(pos) < num_clicks:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                pos.append(pygame.mouse.get_pos())
    return pos


def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


CARD_CM = (8.560, 5.398) # (width, height)

def transform(opts):
    screen, px = setup(opts.image_path)

    print('Click on 4 corners of the card...')
    card_orig = np.float32(loop_clicks(screen, px, num_clicks=4))

    print('Click on 4 points on the step (2 at the bottom, 2 at the top)...')
    steps_orig = np.float32(loop_clicks(screen, px, num_clicks=4))
    
    a, b, c, d = card_orig
    ab = int(np.round(distance(a, b)))
    ac = int(np.round(distance(a, c)))

    card_new = np.float32([a, (a[0]+ab, a[1]), (a[0], a[1]+ac), (a[0]+ab, a[1]+ac)])

    if opts.transform == 'affine':
        M = cv2.getAffineTransform(card_orig[:3], card_new[:3])
    elif opts.transform == 'perspective':
        M = cv2.getPerspectiveTransform(card_orig, card_new)

    img = cv2.imread(opts.image_path)
    width, height, ch = img.shape
    if opts.transform == 'affine':
        output = cv2.warpAffine(img, M,  (height, width))
    elif opts.transform == 'perspective':
        output = cv2.warpPerspective(img, M,  (height, width))

    if opts.transform == 'affine':
        M_pers = np.identity(3)
        M_pers[:2,:3] = M
        M = M_pers
    steps_new = cv2.perspectiveTransform(np.array([steps_orig]), M)[0]

    w, x, y, z = steps_new
    m1 = float(x[1]-w[1])/float(x[0]-w[0])
    m2 = float(z[1]-y[1])/float(z[0]-y[1])
    m = float(m1+m2)/2
    b1 = m*-1*w[0]+w[1]
    b2 = m*-1*y[0]+y[1]

    mp = -1.0 / m
    bp = w[1]-mp*w[0]
    
    tx = float(b2-bp) / (mp-m2)
    ty = mp * tx + bp
    t = [tx, ty]

    step_dist_pixels = distance(w, t)
    card_height_pixels = distance(card_new[0], card_new[2]) 
    step_dist_cm = CARD_CM[1] / card_height_pixels * step_dist_pixels

    print('Step height: %.2f cm' % step_dist_cm) 

    cv2.imshow('image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opts = parse_args()
    transform(opts)

