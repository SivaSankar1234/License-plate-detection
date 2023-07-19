import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pytesseract
from PIL import Image
from matplotlib import cm

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def trim_text(text):
  ans=""
  for i in range(len(text)):
    if(text[i]=='|'and i>0 and i<len(text)-2):
        ans+='I'
    elif(text[i].isalnum() is not True):
      continue
    else:
      ans+=text[i]
  return ans


img = cv2.imread('./test_images/24.jpg')
print("Reading Image.....")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))


bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 100) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img1=img.copy()
cv2.drawContours(img1,keypoints[0],-1,(0,255,0),3)
cv2.imshow("img1",img1)
cv2.waitKey(0)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]


location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 20, True)
    if len(approx) == 4:
        location = approx
        break

#print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))


(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
cv2.imshow("cm",cropped_image)
cv2.waitKey(0)

im = Image.fromarray(np.uint8(cm.gist_earth(cropped_image)*255))


text = pytesseract.image_to_string(im,config ='--psm 6')
text = str.upper(text)
text = trim_text(text)
print(f"License plate number is {text}")

