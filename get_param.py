from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
import cv2
PI = np.pi

IS_PERSPECTIVE = True                               # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])                 # 模型缩放比例
EYE = np.array([0.0, 0.0, 0.3])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 1.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.5, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 720, 480                             # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False                              # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0                             # 考察鼠标位移量时保存的起始位置

MAX_THETA = PI / 3
RVC_THETA = PI /6

Ref_image = cv2.imread('refImage/resize-266.png')

LINE_DIST = 3
X_WIDTH = 0.7
X_OFFSET = 0
# ------------------------------------------------------------------
# ref: https://blog.csdn.net/weixin_38140931/article/details/89214903
def get_lc():
    global RVC_THETA
    if abs(RVC_THETA) < PI / 200:
        return 1000
    elif RVC_THETA < 0:
        return 4 * (1. / math.tan(-RVC_THETA)) / 2.
    else:
        return 4 * (1. / math.tan(RVC_THETA)) / 2.

def get_r():
    global RVC_THETA
    if RVC_THETA != 0:
        return (get_lc() + 1/3) ** 2
    else:
        return 1000

def get_d():
    return 0.5 / 3.

# --------------------------------------------------------------

def get_inside_r():
    global RVC_THETA
    if RVC_THETA != 0:
        return (get_lc() - 1/3) ** 2
    else:
        return 1000

def get_outside_r():
    global RVC_THETA
    if RVC_THETA != 0:
        return (get_lc() + 1/3) ** 2
    else:
        return 1000

def get_line_dist():
    global LINE_DIST
    return LINE_DIST

def get_x_width():
    global X_WIDTH
    return X_WIDTH

def get_x_offset():
    global X_OFFSET
    return X_OFFSET



def drawTorus3(radius, line_width, sides, rings):
    print("%.3f,%.3f"%(get_inside_r(),get_outside_r()))
    width_range = [ i * line_width / sides for i in range(sides) ]
    inside_dest_range = [ get_inside_r() - line_width / 2 + w for w in width_range ]
    outside_dest_range = [ get_outside_r() - line_width / 2 + w for w in width_range ]

    view_theta = get_line_dist() / get_outside_r()
    view_theta_range = [ i * view_theta / rings for i in range(rings) ]
    if RVC_THETA > 0:
        left_xy = [[(math.cos(t) * dist - get_inside_r() - get_x_width() + get_x_offset(), math.sin(t) * dist) for dist in inside_dest_range ] for t in view_theta_range ]
        right_xy = [[(math.cos(t) * dist - get_outside_r() + get_x_width() + get_x_offset(), math.sin(t) * dist) for dist in outside_dest_range ] for t in view_theta_range ]
    else:
        left_xy = [[(math.cos(PI-t) * dist + get_outside_r() - get_x_width() + get_x_offset(), math.sin(PI-t) * dist) for dist in outside_dest_range] for t in view_theta_range ] 
        right_xy = [[(math.cos(PI-t) * dist + get_inside_r() + get_x_width() + get_x_offset(), math.sin(PI-t) * dist) for dist in inside_dest_range] for t in view_theta_range ]

    glColor4f(1.0, 1.0, 1.0, 1.0)
    points_1 = left_xy[0]
    for points in left_xy:
        glBegin(GL_QUAD_STRIP) 
        for i, (x,y) in enumerate(points):
            x1, y1 = points_1[i]
            glVertex3f(x1, y1, 0)
            glVertex3f(x, y, 0)
        glEnd()
        points_1 = points

    points_2 = right_xy[0]
    for points in right_xy:
        glBegin(GL_QUAD_STRIP) 
        for i, (x,y) in enumerate(points):
            x1, y1 = points_2[i]
            glVertex3f(x1, y1, 0)
            glVertex3f(x, y, 0)
        glEnd()
        points_2 = points
    
def save_image():
    glPixelStorei(GL_PACK_ALIGNMENT,4)
    glReadBuffer(GL_FRONT)
    data = glReadPixels(0,0,WIN_W,WIN_H, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes('RGBA', (WIN_W,WIN_H), data)
    image = ImageOps.flip(image)
    #image.save('rvc%.2f.png'%RVC_THETA,'png')
    cvImage = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGBA2BGR)
    print(cvImage.shape)
    mixImage = cv2.add(Ref_image, cvImage)
    cv2.imshow('compare',mixImage)
    cv2.waitKey(1)

def getposture():
    global EYE, LOOK_AT
    
    dist = np.sqrt(np.power((EYE-LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1]-LOOK_AT[1])/dist)
        theta = np.arcsin((EYE[0]-LOOK_AT[0])/(dist*np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0
        
    return dist, phi, theta
    
DIST, PHI, THETA = getposture()                     # 眼睛与观察目标之间的距离、仰角、方位角

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0) # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)          # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)           # 设置深度测试函数（GL_LEQUAL只是选项之一）

    
def reshape(width, height):
    global WIN_W, WIN_H
    
    WIN_W, WIN_H = width, height
    glutPostRedisplay()
    
def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y
    
    '''
    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state==GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()
    '''
    
def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H
    
    '''
    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y
        
        PHI += 2*np.pi*dy/WIN_H
        PHI %= 2*np.pi
        THETA += 2*np.pi*dx/WIN_W
        THETA %= 2*np.pi
        r = DIST*np.cos(PHI)
        
        EYE[1] = DIST*np.sin(PHI)
        EYE[0] = r*np.sin(THETA)
        EYE[2] = r*np.cos(THETA)
            
        if 0.5*np.pi < PHI < 1.5*np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0
        
        glutPostRedisplay()
    '''
    
DIRECT_LEFT = True
def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW
    global RVC_THETA, MAX_THETA
    global DIRECT_LEFT
    global LINE_DIST 
    global X_WIDTH 
    global X_OFFSET

    theta_step = MAX_THETA / 50
    
    '''
    if key == b'x': # 瞄准参考点 x 减小
        LOOK_AT[0] -= 0.01
    elif key == b'X': # 瞄准参考 x 增大
        LOOK_AT[0] += 0.01
    elif key == b'y': # 瞄准参考点 y 减小
        LOOK_AT[1] -= 0.01
    elif key == b'Y': # 瞄准参考点 y 增大
        LOOK_AT[1] += 0.01
    elif key == b'z': # 瞄准参考点 z 减小
        LOOK_AT[2] -= 0.01
        print(LOOK_AT)
    elif key == b'Z': # 瞄准参考点 z 增大
        LOOK_AT[2] += 0.01
        print(LOOK_AT)
    '''

    if key == b'y': # 瞄准参考点 y 减小
        LOOK_AT[1] -= 0.01
    elif key == b'Y': # 瞄准参考点 y 增大
        LOOK_AT[1] += 0.01
    elif key == b'a':
        if DIRECT_LEFT:
            if RVC_THETA >= MAX_THETA:
                DIRECT_LEFT = False
                RVC_THETA = MAX_THETA
            else:
                RVC_THETA = RVC_THETA + theta_step 
        else:
            if RVC_THETA <= -MAX_THETA:
                DIRECT_LEFT = True
                RVC_THETA = -MAX_THETA
            else:
                RVC_THETA = RVC_THETA - theta_step
        print(RVC_THETA)
    elif key == b'x':
        X_OFFSET += 0.01
    elif key == b'X':
        X_OFFSET -= 0.01
    elif key == b'w':
        X_WIDTH += 0.01
    elif key == b'W':
        X_WIDTH -= 0.01
    elif key == b'd':
        LINE_DIST += 0.01
    elif key == b'D':
        LINE_DIST -= 0.01
    elif key == b'q': 
        cv2.destroyAllWindows()
    '''
    elif key == b'\r': # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\x08': # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b' ': # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE 
        glutPostRedisplay()
    '''
    DIST, PHI, THETA = getposture()
    glutPostRedisplay()
    save_image()



def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H
        
    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        
    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
        
    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])
        
    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2], 
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )
    
    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

    # **** 设置后可消除锯齿 *******
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ONE)
    
    drawTorus3(get_r(), 0.05, 5, 100)
    #drawTorus2(get_r(), 0.02, 5, 100)

    # ---------------------------------------------------------------
    glutSwapBuffers()                    # 切换缓冲区，以显示绘制内容

# 摄像头视角
EYE = np.array([0.0, -0.5, 1.0])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 1.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.5, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
'''
#俯视角
EYE = np.array([0.0, 0.0, 2.0])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.5, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
'''

if __name__ == "__main__":
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_MULTISAMPLE
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL')
    
    init()                              # 初始化画布
    glutDisplayFunc(draw)               # 注册回调函数draw()
    glutReshapeFunc(reshape)            # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)           # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)         # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)           # 注册键盘输入的函数keydown()
    cv2.namedWindow('compare')
    
    glutMainLoop()                      # 进入glut主循环
