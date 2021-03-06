import cv2
from os.path import isfile, join
from os import listdir

class CV_Prepare:
    def __init__(self):
        self.dir = "/home/wangheng/Downloads/资料下载/images/"
        self.files = [f for f in listdir(self.dir)]

    def read_file(self, file_path):
        image = cv2.imread(file_path)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_scale, (3,3), 0)
        egge = cv2.Canny(gaussian_blur, 150, 50)

        cv2.imshow("image", image)
        cv2.imshow("gaussion_blur_image", gaussian_blur)
        cv2.imshow("eege", egge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cv_prepare = CV_Prepare()
    cv_prepare.read_file(cv_prepare.dir + cv_prepare.files[0])