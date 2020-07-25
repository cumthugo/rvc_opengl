from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
PI = np.pi

IS_PERSPECTIVE = True                               # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])                 # 模型缩放比例
EYE = np.array([0.0, 0.0, 0.3])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 1.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.5, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 1080, 720                             # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False                              # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0                             # 考察鼠标位移量时保存的起始位置

MAX_THETA = PI / 5

RVC_THETA = 1.97758476261356e-16 #MAX_THETA

# ------------------------------------------------------------------
# ref: https://blog.csdn.net/weixin_38140931/article/details/89214903
def get_lc():
    global RVC_THETA
    if abs(RVC_THETA) < PI / 200:
        return 1000
    elif RVC_THETA < 0:
        return 2 * (1. / math.tan(-RVC_THETA)) / 3.
    else:
        return 2 * (1. / math.tan(RVC_THETA)) / 3.

def get_r():
    global RVC_THETA
    if RVC_THETA != 0:
        return (get_lc() + 1/3) ** 2
    else:
        return 1000

def get_d():
    return 0.5 / 3.
# --------------------------------------------------------------

# -------绘制圆环
def drawTorus(radius, tube_radius, sides, rings):
    side_delta = tube_radius / sides
    ring_delta = (1.3 / get_r()) / rings #以3m标么化, 1.3 约等于4m左右
    theta = 0.0 if RVC_THETA >= 0 else PI
    cosTheta = 1.0
    sinTheta = 0.0

    glColor4f(1.0, 1.0, 1.0, 1.0)
    for i in range(rings):
        theta1 = (theta + ring_delta) if RVC_THETA >= 0 else (theta - ring_delta)
        cosTheta1 = math.cos(theta1)
        sinTheta1 = math.sin(theta1)
        
        glBegin(GL_QUAD_STRIP) #左边线
        phi = 0.0
        for j in range(sides * 2):
            phi = phi + side_delta
            dist = radius - tube_radius/2 + phi

            if abs(RVC_THETA) < PI / 200:    # 接近 0 度，特殊处理
                glVertex3f(dist - radius - 0.3, i * 0.013 , 0)
                glVertex3f(dist - radius - 0.3, (i+1) * 0.013 , 0)
            elif RVC_THETA > 0: # 左边圆
                glVertex3f(cosTheta * dist - get_r() - 0.3, sinTheta * dist , 0)
                glVertex3f(cosTheta1 * dist - get_r() - 0.3, sinTheta1 * dist, 0)
            else: #右边圆
                glVertex3f(cosTheta * dist + get_r() - 0.3 + 0.025, sinTheta * dist , 0)
                glVertex3f(cosTheta1 * dist + get_r() - 0.3 + 0.025, sinTheta1 * dist, 0)

        glEnd()

        glBegin(GL_QUAD_STRIP) #右边线
        phi = 0.0
        for j in range(sides * 2):
            phi = phi + side_delta
            dist = radius - tube_radius/2 + phi
            
            if abs(RVC_THETA) < PI / 200:
                glVertex3f(dist - radius + 0.3, i * 0.013 , 0)
                glVertex3f(dist - radius + 0.3, (i+1) * 0.013 , 0)
            elif RVC_THETA > 0:
                glVertex3f(cosTheta * dist - get_r() + 0.3, sinTheta * dist , 0)
                glVertex3f(cosTheta1 * dist - get_r() + 0.3, sinTheta1 * dist, 0)
            else:
                glVertex3f(cosTheta * dist + get_r() + 0.30 + 0.025, sinTheta * dist , 0)
                glVertex3f(cosTheta1 * dist + get_r() + 0.30 + 0.025, sinTheta1 * dist, 0)

        glEnd()
        theta = theta1
        cosTheta = cosTheta1
        sinTheta = sinTheta1

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
    
    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state==GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()
    
def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H
    
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
    
DIRECT_LEFT = True
def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW
    global RVC_THETA, MAX_THETA
    global DIRECT_LEFT

    theta_step = MAX_THETA / 50
    
    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z', b'r', b'l',b'a']:
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
        elif key == b'r':
            RVC_THETA = RVC_THETA - theta_step if RVC_THETA > -MAX_THETA else -MAX_THETA
            print(RVC_THETA)
        elif key == b'l':
            RVC_THETA = RVC_THETA + theta_step if RVC_THETA < MAX_THETA else MAX_THETA
            print(RVC_THETA)
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


        
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
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
    
    '''
    # ---------------------------------------------------------------
    glBegin(GL_LINES)                    # 开始绘制线段（世界坐标系）
    
    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)        # 设置当前颜色为红色不透明
    glVertex3f(-0.8, 0.0, 0.0)           # 设置x轴顶点（x轴负方向）
    glVertex3f(0.8, 0.0, 0.0)            # 设置x轴顶点（x轴正方向）
    
    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)        # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -0.8, 0.0)           # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, 0.8, 0.0)            # 设置y轴顶点（y轴正方向）
    
    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)        # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -0.8)           # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, 0.8)            # 设置z轴顶点（z轴正方向）
    
    glEnd()                              # 结束绘制线段
    '''
    
    drawTorus(get_r(), 0.02, 5, 100)

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
    
    print(get_lc())
    glutMainLoop()                      # 进入glut主循环
