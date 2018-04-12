#!/usr/bin/env python
# -*- coding: utf-8 -*-
import objc, Quartz
from AppKit import NSBitmapImageRep
from Quartz.CoreGraphics import CGMainDisplayID

from cv2 import cv2

from PIL import ImageGrab

import numpy as np

from pymouse import PyMouse
from pykeyboard import PyKeyboard

from time import sleep
from copy import deepcopy

# Input init
# Ref: https://github.com/PyUserInput/PyUserInput
# Usage:
# m.click(0, 0, 1)  使用 X11 坐标
# k.type_string('Hello, World!')
m = PyMouse()
k = PyKeyboard()

# Objc bridge init
# Ref: https://github.com/StoneMoe/pymaScreen
objc.parseBridgeSupport("""<?xml version='1.0'?>
<!DOCTYPE signatures SYSTEM "file://localhost/System/Library/DTDs/BridgeSupport.dtd">
<signatures version='1.0'>
  <depends_on path='/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation' />
  <depends_on path='/System/Library/Frameworks/IOKit.framework/IOKit' />
  <depends_on path='/System/Library/Frameworks/CoreServices.framework/CoreServices' />
  <function name='CGDisplayCreateImageForRect'>
    <retval already_cfretained='true' type='^{CGImage=}' />
    <arg type='I' />
    <arg type='{CGRect={CGPoint=ff}{CGSize=ff}}' type64='{CGRect={CGPoint=dd}{CGSize=dd}}' />
  </function>
</signatures>
""", globals(), '/System/Library/Frameworks/ApplicationServices.framework/Frameworks/CoreGraphics.framework')
mainID = CGMainDisplayID()

# Screen detection
print('[加载]屏幕参数')
x11_x, x11_y = m.screen_size()
pil_x, pil_y = ImageGrab.grab().size
DPI_times = pil_x / x11_x
EMULATOR_HEADER = 39
EMULATOR_BOTTOM = 42
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', 500, 500)
cv2.moveWindow('output', int(x11_x) - 500, int(x11_y) - 500)
print('[加载]X11 size:', (x11_x, x11_y))
print('[加载]PIL size:', ImageGrab.grab().size)
print('[加载]DPI times:', DPI_times)


# Helpers - 图像处理
screen_data = None

class ScanIter(object):
    def __init__(self, startp, endp):
        self.start_x, self.start_y = startp
        self.end_x, self.end_y = endp

    def __iter__(self):
        for y in range(self.start_y, self.end_y + 1):
            for x in range(self.start_x, self.end_x + 1):
                yield x, y


class LocatedImgX11Coordinates(object):
    def __init__(self, start_p, center_p, end_p):
        self.start_point = start_p
        self.center_point = center_p
        self.end_point = end_p


def get_color(x, y):
    """
    取某一点颜色
    :param x: x, 使用 X11 坐标
    :param y: y, 使用 X11 坐标
    :return:
    """
    image = CGDisplayCreateImageForRect(mainID, ((0, 0), (x11_x, x11_y)))
    bitmap = NSBitmapImageRep.alloc()
    bitmap.initWithCGImage_(image)
    data = str(bitmap.colorAtX_y_(x * DPI_times, y * DPI_times))
    arr = data.split(' ')
    return int(float(arr[1]) * 255), int(float(arr[2]) * 255), int(float(arr[3]) * 255)


def get_color_pil(x, y):
    screen = ImageGrab.grab()
    screen = screen.load()
    return screen[x * DPI_times, y * DPI_times]


def find_color(rgb, limit_area=False):
    """
    在整个屏幕 / 指定区域找到某个颜色 TODO
    :param rgb:
    :param limit_area: 从 (0, 0) 到 (x, y) 的区域, 使用 X11 坐标
    :type limit_area: tuple
    :return: 颜色位置
    """
    r, g, b = rgb
    if limit_area is False:
        image = CGDisplayCreateImageForRect(mainID, ((0, 0), (x11_x, x11_y)))
    else:
        image = CGDisplayCreateImageForRect(mainID, ((0, 0), (limit_area[0], limit_area[1])))
    bitmap = NSBitmapImageRep.alloc()
    bitmap.initWithCGImage_(image)


def update_screen_data():
    global screen_data
    pil_img = ImageGrab.grab().convert('RGB')
    screen_data = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def locate_img(filename, area, need_score=None, show=True, use_result=False):
    """
    匹配指定图像并返回坐标, 匹配前先 update_screen_data()
    :param filename:
    :param area: 使用 X11 坐标 (left, upper, right, lower)
    :param debug:
    :return: X11 坐标, (x, y, x_center, y_center, x_end, y_end)
    """
    result_valid = False
    font_color = (0, 0, 255)
    offset_x = 0
    offset_y = 0
    if use_result:
        offset_x = area[0] + emulator_area[0]
        offset_y = area[1] + emulator_area[1]
    # X11 size to PIL Size
    area = (area[0] * DPI_times, area[1] * DPI_times, area[2] * DPI_times, area[3] * DPI_times)
    method = cv2.TM_SQDIFF_NORMED
    small_image = cv2.imread(filename)
    large_image = deepcopy(screen_data)[int(area[1]):int(area[3]), int(area[0]):int(area[2])]

    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    mn, mx, mnLoc, _ = cv2.minMaxLoc(result)

    # 取合适的 threshold
    if need_score is None:
        need_score = 0.2

    if mn > need_score:
        print('[CV]Mismatch: min %s | max %s | need_score %s | %s' % (mn, mx, need_score, filename))
    else:
        print('[CV]Matched: min %s | max %s | need_score %s | %s' % (mn, mx, need_score, filename))
        result_valid = True
        font_color = (0, 255, 0)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx, MPy = mnLoc
    # Extract target size
    trows, tcols = small_image.shape[:2]

    # Draw rect on larger image and put text
    cv2.rectangle(large_image, (MPx, MPy), (MPx + tcols, MPy + trows), font_color, 2)
    cv2.putText(large_image, 'Finding ' + filename.split('.')[0], (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)
    # Display it
    cv2.imshow('output', large_image)
    cv2.waitKey(1)

    if result_valid:
        # PIL coordinates to X11 coordinates, and plus area offset
        MPx = MPx / DPI_times + offset_x
        MPy = MPy / DPI_times + offset_y
        return LocatedImgX11Coordinates((MPx, MPy),
                                        (MPx + tcols / DPI_times / 2, MPy + trows / DPI_times / 2),
                                        (MPx + tcols / DPI_times, MPy + trows / DPI_times))
    else:
        return None


def color_cmp(a, b):
    diff_sum = 0
    diff_sum += abs(a[0] - b[0])
    diff_sum += abs(a[1] - b[1])
    diff_sum += abs(a[2] - b[2])
    return diff_sum


def mouse_left(x, y, relative=True):
    """
    鼠标点击
    :param x: x11
    :param y: x11
    :param relative: 是否将坐标区域限制映射到模拟器画面中
    :return:
    """
    if relative:
        x += emulator_area[0]
        y += emulator_area[1]
    m.click(x, y, 1)
    sleep(0.5)
    update_screen_data()


def area_offset(left, upper, right, lower):
    return emulator_area[0] + left, emulator_area[1] + upper, emulator_area[2] - right, emulator_area[3] - lower


# Helpers - 游戏流程
def is_in_instance_map():
    # 是否在副本里 Optimized
    print('[检查]是否在副本中')
    if locate_img('taskmap_exit_button.png', area_offset(930, 120, 180, 400), need_score=0.2):
        print('[状态]正在副本中')
        return True
    else:
        print('[状态]不在副本中')
        return False


def is_in_battle_result():
    print('[检查]是否在结算界面')
    if locate_img('battle_result.png', area_offset(560, 0, 0, 300), need_score=0.2):
        print('[状态]正在结算界面')
        return True
    else:
        print('[状态]不在结算界面中')
        return False


def is_in_main_page():
    # 是否在主界面 Optimized
    print('[检查]是否在主界面中')
    if locate_img('main_page.png', task_area):
        print('[状态]正在主界面中')
        return True
    else:
        print('[状态]不在主界面中')
        return False


def is_energy_enough():
    # 检查体力
    print('[检查]体力状态')
    if not locate_img('energy_not_enough.png', emulator_area, need_score=0.2):
        print('[状态]体力充足')
        return True
    else:
        print('[状态]体力不足')
        return False


def is_combine_skill_ready():
    # 合体技是否准备好
    print('[检查]合体技状态')
    if locate_img('combine_skill_ok.png', area_offset(640, 260, 220, 190), need_score=0.08):
        print('[状态]合体技准备就绪')
        return True
    else:
        print('[状态]合体技未准备好')
        return False


def skip_battle_result():
    # 跳过副本结算画面
    print('[操作]跳过副本结算画面')
    mouse_left(967, 550)
    mouse_left(967, 550)
    mouse_left(967, 550)
    mouse_left(967, 550)


def is_in_detail_page():
    # 是否在详情页 Optimized
    print('[检查]是否在详情页中')
    button = locate_img('close_detail_button.png', area_offset(930, 0, 0, 500), need_score=0.1)
    button2 = locate_img('close_dialog_button.png', area_offset(930, 0, 0, 500), need_score=0.05)
    if button is not None or button2 is not None:
        print('[操作]在详情页/Dialog中')
        return True
    else:
        print('[状态]不在详情页/Dialog中')
        return False


def is_in_store():
    # 是否在商店页
    print('[检查]是否在商店页中')
    button = locate_img('store_buy_button.png', area_offset(670, 340, 0, 100), need_score=0.1)
    if button is not None:
        print('[操作]在商店页中')
        return True
    else:
        print('[状态]不在商店页中')
        return False


def has_action_button():
    # 互动按钮
    print('[检查]是否存在互动按钮')
    button = locate_img('action_button.png', area_offset(655, 260, 280, 210), need_score=0.01)
    button2 = locate_img('action_button2.png', area_offset(655, 260, 280, 210), need_score=0.01)
    button3 = locate_img('action_button3.png', area_offset(655, 260, 280, 210), need_score=0.01)
    if button is not None or button2 is not None or button3 is not None:
        print('[状态]存在互动按钮')
        return True
    else:
        print('[状态]无互动按钮')
        return False


def has_chat_skip_button():
    # 剧情跳过按钮 Optimized
    print('[检查]是否存在剧情跳过按钮')
    button = locate_img('chat_skip_button.png', area_offset(920, 0, 0, 510), need_score=0.005)
    if button is not None:
        print('[状态]存在跳过按钮')
        return True
    else:
        print('[状态]无跳过按钮')
        return False


def is_in_chat():
    # 是否在对话页面 Optimized
    print('[检查]是否在对话页面中')
    button = locate_img('chat_page.png', area_offset(0, 400, 950, 0))
    if button is not None:
        print('[操作]在对话页面中')
        return True
    else:
        print('[状态]不在对话页面中')
        return False


def is_in_party_page():
    # 是否在队伍页面 Optimized
    print('[检查]是否在队伍页面')
    title = locate_img('party_page.png', area_offset(0, 0, 870, 590))
    if title is not None:
        print('[操作]在队伍页面中')
        return True
    else:
        print('[状态]不在队伍页面中')
        return False


def use_close_button():
    # 关闭按钮
    print('[操作]点击右上角关闭按钮')
    mouse_left(1125, 30)


def enter_dog_feed_instance():
    # 进入狗粮本
    print('[操作]进入狗粮本')
    mouse_left(909, 186)  # 活动
    mouse_left(1135, 340)  # 卡牌
    mouse_left(786, 360)  # 狗粮本
    mouse_left(970, 600)  # 进入 / 创建队伍
    if is_in_party_page():
        mouse_left(737, 610)  # 取消自动匹配
        mouse_left(930, 610)  # 进入
    mouse_left(720, 450)  # 需要电脑


def enter_event_normal():
    # 活动-通常任务 TODO
    print('[操作]进入通常活动')
    mouse_left(788, 226)  # 活动
    mouse_left(660, 155)  # 收起限时任务
    mouse_left(670, 320)  # 第一个日常


def get_daily_delegate():
    # 活动-委托-日常
    print('[操作]领取日常委托')
    mouse_left(788, 226)  # 活动
    mouse_left(983, 588)  # 委托
    for n in range(7):
        mouse_left(174, 341)  # 第一个委托焦点
        mouse_left(174, 535)  # 领取任务
        m.move(883, 371)  # 移动，准备拖拽
        m.drag(539, 355)  # 向左拖拽一个任务
        mouse_left(174, 341)  # 第一个委托焦点
        sleep(0.5)


def auto_task_target():
    # 自动寻路 Optimized
    # 仅选择日常和主线
    print('[操作]尝试从任务列表自动寻路 - 仅 日常/主线')
    tag = locate_img('main_task_tag.png', area_offset(860, 0, 0, 350), use_result=True)
    tag2 = locate_img('daily_task_tag.png', area_offset(860, 0, 0, 350), use_result=True)
    if tag:
        mouse_left(*tag.center_point)
    elif tag2:
        mouse_left(*tag2.center_point)


def use_energy_drink():
    print('[操作]使用体力药水')
    mouse_left(520, 420)  # 需要电脑


def use_combine_skill():
    print('[操作]使用合体技')
    mouse_left(805, 370)


def use_action_button():
    print('[操作]使用交互按钮')
    mouse_left(705, 320)


def use_chat_button():
    print('[操作]尝试选择对话')
    button = locate_img('correct_chat_choice.png', emulator_area, use_result=True)
    if button:
        print('[操作]找到了更加有效的对话选项')
        mouse_left(*button.center_point)
    else:
        print('[操作]未找到更加有效的对话选项，选择默认项')
        mouse_left(810, 300)


# print(get_rgb(69, 1016))
# 主线等级不足字体颜色 211 121 51


# m.move(985, 96)
# print(get_color(985, 96))

# 定位模拟器
print('[加载]模拟器位置')
update_screen_data()
start_x, start_y = locate_img('mumu_locate.png', (0, 0, x11_x, x11_y), show=False).start_point
end_x, end_y = locate_img('mumu_locate_end.png', (0, 0, x11_x, x11_y), show=False).end_point
emulator_area = (start_x, start_y + EMULATOR_HEADER, end_x, end_y - EMULATOR_BOTTOM)
task_area = area_offset(860, 0, 0, 350)
print('[加载]定位完毕', emulator_area)

disable_dog_feed = True
enable_dog_feed_counter = 0

daily_delegate_got = True

while True:
    print('===========[轮询]============')
    update_screen_data()
    if is_in_instance_map():  # 是否在副本中
        if is_combine_skill_ready():
            use_combine_skill()
            continue
    elif is_in_chat():  # 在对话界面
        if has_chat_skip_button():  # 检查快进对话按钮
            mouse_left(1120, 20)  # 点击快进
        else:
            use_chat_button()  # 点击对话选项
    # elif is_in_battle_result():
    #    skip_battle_result()
    elif is_in_detail_page():  # 在任何的详情页
        while is_in_detail_page():  # 退出直到不在详情页
            use_close_button()
    elif is_in_main_page():  # 是否在主页面
        if not disable_dog_feed:  # 尝试进入狗粮本
            enter_dog_feed_instance()
            if not is_energy_enough():
                use_energy_drink()
                if is_in_store():
                    # 没体力药了，停止进入狗粮本
                    disable_dog_feed = True
                    use_close_button()
                    continue
            continue
        else:  # 计数器恢复进入狗粮本
            enable_dog_feed_counter += 1
            if enable_dog_feed_counter > 20:
                enable_dog_feed_counter = 0
                disable_dog_feed = False
        if not daily_delegate_got:  # 接日常委托任务
            get_daily_delegate()
            daily_delegate_got = True
            continue
        if has_action_button():  # 检查交互按钮
            use_action_button()
            sleep(6)
            continue
        auto_task_target()  # 点击任务列表自动寻路
        sleep(5)
    else:
        print('[[[[[[[[不支持的页面状态]]]]]]]')
        continue
