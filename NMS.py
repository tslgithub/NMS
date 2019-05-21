# -*- coding: utf-8 -*-
# @Author  : tslgithub
# @Email   : mymailwith163@163.com
# @Time    : 19-5-9 上午10:55
# @File    : NMS.py
# @Software: PyCharm

import numpy as np
import cv2,copy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def write(img,name):
    cv2.imwrite('tmp/'+str(name)+'.jpg',img)

def Box():

    ax,ay = 0,0
    step_pix = 5
    step_pix_B = 10
    size = 1000
    size_plt = 500
    aw,ah = 400,400
    bw,bh = 100,100

    fps = 1
    video_size = (size, size)
    videowriter = cv2.VideoWriter("tmp/a.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, video_size)

    back_tmp = np.ones((size,size),dtype='uint8')*255
    write(back_tmp,'back')
    back_tmp = cv2.imread('tmp/back.jpg')
    back_tmp = cv2.resize(back_tmp,(size,size))
    x,y,x_line,y_line =[], [],[0],[0]
    y_line2 = [0]

    back_plt = cv2.resize(back_tmp,(size,size_plt))
    back_plt_no_NMS = cv2.resize(back_tmp,(size,size_plt))
    # back_plt = cv2.threshold(back_plt,127,255,cv2.THRESH_BINARY_INV)[-1]
    for s in range(1,int(size/step_pix)):
        new_ax = ax+(s-1)*step_pix
        new_ay = ay+(s-1)*step_pix

        new_bx = size - (ax+(s-1)*step_pix_B)-bw
        new_by = size - (ay+(s-1)*step_pix_B)-bh

        back = copy.copy(back_tmp)

        cx = new_bx
        cy = new_by
        cw = new_ax+aw-(new_bx)
        ch = new_ay+aw-(new_by)

        if cw>bw and ch>bh:
            cw = bw
            ch = bh
        if cx<new_ax and cy<new_ay:
            cx = new_ax
            cy = new_ay

            cw = new_bx+bw-new_ax
            ch = new_by+bh-new_ay

        size_a = aw * ah
        size_b = bw * bh
        if cw<0 or ch<0:
            cw = 0
            min_size = 0
        else:
            min_size = min(size_a,size_b)

        # size_c = cw*ch
        # IOU = size_c/min(size_a,size_b)
        # IOU2 = size_c/(size_a+size_b-size_c)
        # print(IOU,IOU2)
        # x.append(s)
        # y.append(size_c/10000)
        #
        # x_line.append(step_pix *s )
        # y_line.append(int(size_c/100))
        #
        # color = (0,0,0)
        # if y_line[s]>50:
        #     color = (0,0,255)
        #     # c_color = []
        #
        # cv2.line(back_plt,       (x_line[s-1], size_plt-y_line[s-1]),(x_line[s],size_plt-y_line[s]),color,5 )
        # cv2.line(back_plt_no_NMS,(x_line[s-1], size_plt-y_line[s-1]),(x_line[s],size_plt-y_line[s]),color,5 )

        size_c = cw * ch
        IOU = size_c / min(size_a, size_b)
        IOU2 = size_c / (size_a + size_b - size_c)
        print(IOU, IOU2)
        # x.append(s)
        # y.append(IOU*10)

        x_line.append(step_pix * s)
        y_line.append(int(IOU*size_plt*0.7))
        y_line2.append(int(IOU2*size_plt*0.7))

        color = (0, 0, 0)
        color_no_NMS = (0, 0, 0)
        # if y_line[s] > 50:

        if IOU >= 0.5:
            color = (0, 0, 255)
        if IOU2 >= 0.5:
            color_no_NMS = (0, 0, 255)

        cv2.line(back_plt, (x_line[s - 1], size_plt - y_line[s - 1]), (x_line[s], size_plt - y_line[s]), color, 5)
        cv2.line(back_plt_no_NMS, (x_line[s - 1], size_plt - y_line2[s - 1]), (x_line[s], size_plt - y_line2[s]), color_no_NMS,5)

        nms_line = np.append(back_plt[:,slice(0,int(size/2))],back_plt_no_NMS[:,slice(0,int(size/2))],axis=1)
        # back.shape[::-1], nms_line.shape[::-1]
        write(nms_line,'m_'+str(s))

        cv2.putText(back_plt,'IOU_WITH_NMS',     (0,50),cv2.FONT_HERSHEY_PLAIN,3,color,2)
        cv2.putText(back_plt_no_NMS,'IOU_NO_NMS',(0,50),cv2.FONT_HERSHEY_PLAIN,3,color,2)
        # back[slice(size - size_plt, size), slice(0, int(size * 5 / 12))] = back_plt[:, slice(0, int(size * 5 / 12))]

        back = np.append(back,nms_line,axis=0)

        pil_im = Image.fromarray(back)
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 50, encoding="utf-8")
        # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
        draw = ImageDraw.Draw(pil_im)
        draw.text((0, size-100), "红色区域：触发ＮＭＳ", (0, 0, 255), font=font)
        back = np.array(pil_im)

        if ch>0 and cw > 0:
            box_A = cv2.rectangle(back,
                                  (new_ax, new_ay),
                                  (new_ax + aw, new_ay + ah),
                                  (255, 0, 0), 2)
            cv2.putText(back, 'A', (new_ax, new_ay), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 2)

            box_B = cv2.rectangle(back,
                                  (new_bx, new_by),
                                  (new_bx + bw, new_by + bh),
                                  (0, 255, 0), 2)
            cv2.putText(back, 'B', (new_bx, new_by), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 2)

            c_color = [0,0,0]
            if IOU>0.5:
                c_color = [0,0,255]
            back[slice(cy,cy+ch),slice(cx,cx+cw)] = c_color
            cv2.putText(back, 'C', (cx+int(cw/2), cy+int(ch/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        else:
            box_A = cv2.rectangle(back,
                                  (new_ax, new_ay),
                                  (new_ax + aw, new_ay + ah),
                                  (255, 0, 0), 2)
            cv2.putText(back, 'A', (new_ax, new_ay), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 2)

            box_B = cv2.rectangle(back,
                                  (new_bx, new_by),
                                  (new_bx + bw, new_by + bh),
                                  (0, 255, 0), 2)
            cv2.putText(back, 'B', (new_bx, new_by), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 2)
            # write(back, 1)

        write(back_plt, 4)
        # cv2.putText(back_plt,'IOU')

        # np.append(back,back_plt,axis=2)
        cv2.imwrite('tmp/step/'+str(s)+'.jpg',back)
        write(back,1)
        videowriter.write(back)
        print(s)

    plt.plot(x,y)
    plt.ylabel('IOU')
    plt.xlabel('step')

    plt.savefig('tmp/result.jpg')
    # plt.show()


def main():
    Box()

if __name__ == '__main__':
    main()
