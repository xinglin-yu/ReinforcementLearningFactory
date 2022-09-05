import cv2
from matplotlib import animation
import matplotlib.pyplot as plt
import random

from typing import List


class ImageClient:

    @staticmethod
    def get_arrow(action, cx, cy, half_len):
        """
        :param action:
        :param cx:
        :param cy:
        :param half_len: the half length of arrow
        :return:
        """
        if action == 0:  # left
            return (cx + half_len, cy), (cx - half_len, cy)
        if action == 1:  # down
            return (cx, cy - half_len), (cx, cy + half_len)
        if action == 2:  # right
            return (cx - half_len, cy), (cx + half_len, cy)
        if action == 3:  # up
            return (cx, cy + half_len), (cx, cy - half_len)

    @staticmethod
    def add_arrow(policy, order, in_img, out_img, render=True, offset=(80, 0)):
        """
        The in_img is 640*480. The real frozen lake is 480*480,
        So the blank width is (640-480)/2=80 in left/right in x direction
        :param policy:
        :param order: the order of the frozen lake
        :param infile: in file name
        :param outfile: out file name
        :return:
        """
        img = cv2.imread(in_img)  # 读取图像

        img_width, img_height = img.shape[1], img.shape[0]
        cell_width = img_height / order

        for i in range(order):
            for j in range(order):
                cell_cx = offset[0] + cell_width / 2 + cell_width * j
                cell_cy = offset[1] + cell_width / 2 + cell_width * i

                start, end = ImageClient.get_arrow(int(policy[i * order + j]), int(cell_cx), int(cell_cy),
                                                   half_len=int(0.25 * cell_width))

                # cv2.arrowedLine( 输入图像，起始点(x,y)，结束点(x,y)，线段颜色，线段厚度，线段样式，位移因数， 箭头因数)
                img = cv2.arrowedLine(img, start, end, (0, 0, 255), 2, 8, 0, 0.3)

        cv2.imwrite(filename=out_img, img=img)

        if render:
            cv2.imshow("img", img)
            # cv2.waitKey(0)

    @staticmethod
    def add_path(paths: List[List[int]], order, in_img, out_img, episode_terminate_points,
                 render=True, offset=(80, 0)):
        """
        The in_img is 640*480. The real frozen lake is 480*480,
        So the blank width is (640-480)/2=80 in left/right in x direction
        :param paths: the possible paths
        :param order: the order of the frozen lake
        :param infile: in file name
        :param outfile: out file name
        :return:
        """
        img = cv2.imread(in_img)  # 读取图像

        img_width, img_height = img.shape[1], img.shape[0]
        cell_width = img_height / order
        legend_color = (0, 0, 255)

        for path in paths:
            # 随机生成颜色
            color = tuple([random.randint(100, 255) for _ in range(3)])
            for i in range(1, len(path)):
                # 起点
                x, y = path[i - 1] // order, path[i - 1] % order
                start = (int(offset[0] + cell_width / 2 + cell_width * y),
                         int(offset[1] + cell_width / 2 + cell_width * x))

                # 终点
                x, y = path[i] // order, path[i] % order
                end = (int(offset[0] + cell_width / 2 + cell_width * y),
                       int(offset[1] + cell_width / 2 + cell_width * x))

                # cv2.arrowedLine( 输入图像，起始点(x,y)，结束点(x,y)，线段颜色，线段厚度，线段样式，位移因数， 箭头因数)
                img = cv2.arrowedLine(img, start, end, color, thickness=2, line_type=None, shift=None, tipLength=None)

            # 添加end state
            x, y = path[-1] // order, path[-1] % order
            end = (int(offset[0] + cell_width / 2 + cell_width * y),
                   int(offset[1] + cell_width / 2 + cell_width * x))
            cv2.putText(img,
                        text=f"S{path[-1]}",
                        org=end,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=legend_color,
                        thickness=2)

        # 添加legend, img_height + int(offset[0]+5)
        cv2.putText(img,
                    text=f"{len(paths)} Paths",
                    org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=legend_color,  # black
                    thickness=1)
        num = 1
        for key, value in episode_terminate_points.items():
            cv2.putText(img,
                        text=f"S{key}:{value}",
                        org=(5, 50 + 30 * num),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=legend_color,  # black
                        thickness=1)
            num += 1

        cv2.imwrite(filename=out_img, img=img)

        if render:
            cv2.imshow("img", img)
            # cv2.waitKey(0)

    @staticmethod
    def save_frame(frame, filename):
        plt.imshow(frame)
        plt.axis('off')

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # the saved img is 640*480
        plt.savefig(filename)

    @staticmethod
    def save_gif(frames, filename):
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(filename, writer='imagemagick', fps=30)


if __name__ == '__main__':
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    episode_terminate_points = {15: 1.0}
    ImageClient.add_arrow(policy, order=4,
                          in_img='../Doc/Snapshot/FrozenLake4x4.png',
                          out_img='../Doc/Snapshot/FrozenLake4x4_Policy.png')
