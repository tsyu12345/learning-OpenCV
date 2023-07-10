"""_summary_

openCVによる、丸つき数字の認識サンプル
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_circles(img_path: str) -> None:
    #円の検出サンプル
    """cv2.HoughCircles()パラメーターの説明
        img: 入力画像
        method: 検出手法
        dp: 検出精度 値が大きいほど検出基準が緩くなり、値が小さいほど検出基準が厳しく
        minDist:検出される円同士が最低限離れていなければならない距離
        param1: 円検出におけるCannyエッジ検出の閾値(上限値)の設定、上限値以下, 下限値(paramの1/2)以上のものを正しいエッジとして判断.
        param2: 円の中心を検出する際の閾値、低い値にすると円の誤検出が多くなり、高い値にすると未検出が多くなる。
        minRadius: 検出する円の最小半径
        maxRadius: 検出する円の最大半径
    """

    image = cv2.imread(img_path)
    print('load image')
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray_scale,
        cv2.HOUGH_GRADIENT,
        dp=0.2,
        minDist=5,
        minRadius=3
    )
    print("detecting circles")

    #cv2.circle関数のいくつかの引数は Int型である必要があるので、
    #このタイミングで np.float型から整数値に丸め、さらに16ビット情報にキャストします
    circles = np.uint16(np.around(circles))

    #検出した円を描画する:
    # circlesには、検出した円の数だけの座標配列が格納されているので、これをfor文で回して描画する
    for i in circles[0, :]:
        # 円を描画する
        # cv2.circle(画像, 円の中心座標, 円の半径, 色, 線の太さ)
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # 円の中心を描画する
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    plot_circles(image)

def plot_circles(image: np.ndarray) -> None:
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.imshow(image_rgb)
    plt.show()



def main():

    detect_circles('sheet1.jpg')


    

    




if __name__ == '__main__':
    main()