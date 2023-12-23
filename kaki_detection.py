import numpy as np
import cv2
from scipy.optimize import curve_fit

IMAGE_PATH = "./img/kaki.JPG"
OUTPUT_MASKED_IMAGE_PATH = './img/masked_kaki.jpg'
OUTPUT_CONTOURED_IMAGE_PATH = './img/contoured_kaki.jpg'


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# 画像を読み込み
kaki_img = cv2.imread(IMAGE_PATH)
# 画像をHSVに変換
kaki_HSV = cv2.cvtColor(kaki_img, cv2.COLOR_BGR2HSV)
# 中央値フィルタを使用してノイズを減らす
kaki_blur = cv2.medianBlur(kaki_HSV, 21)

# オレンジ色のHSV範囲を調整
lower_orange = np.array([5, 130, 150])
upper_orange = np.array([30, 255, 255])

# HSV画像からオレンジ色の部分のみをマスクとして取得
mask_orange = cv2.inRange(kaki_blur, lower_orange, upper_orange)

# 膨張処理をマスクに適用
kernel = np.ones((5, 5), np.uint8)
mask_orange_dilated = cv2.dilate(mask_orange, kernel, iterations=2)

# マスクを元の画像に適用してオレンジ色の部分のみを抽出
kaki_masked = cv2.bitwise_and(kaki_img, kaki_img, mask=mask_orange_dilated)

# マスクされた画像をファイルに書き出す
cv2.imwrite(OUTPUT_MASKED_IMAGE_PATH, kaki_masked)

# マスクされた画像に輪郭を描く
kaki_contoured = kaki_img.copy()

# マスクから輪郭を見つける
contours, _ = cv2.findContours(mask_orange_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭の面積を計算してリストに追加
areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 400]

# ヒストグラムを生成するために面積のデータを使用
area_hist, bin_edges = np.histogram(areas, bins='auto')
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 最小二乗法を用いてガウス関数にフィット
popt, _ = curve_fit(gaussian, bin_centers, area_hist, p0=[1, np.mean(areas), np.std(areas)])
common_area = popt[1]  # 最も一般的な面積のビンの中心

# 最も一般的な面積を用いて個数を推定
total_area = sum(areas)
estimated_count = total_area / common_area if common_area > 0 else 0

print(f"最小二乗法による共通面積を使用した柿の推定個数: {int(estimated_count)}")

# 輪郭を元の画像に描画
cv2.drawContours(kaki_contoured, contours, -1, (255, 255, 255), 1)

# 輪郭が描かれた画像をファイルに書き出す
cv2.imwrite(OUTPUT_CONTOURED_IMAGE_PATH, kaki_contoured)
