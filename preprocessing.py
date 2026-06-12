""" I want 5 classes for 5 preprocessing steps 
      1. remove black borders(is this even good?)
      Resize(224,224) -> to use Vit
      3. Contrast enhancement CLAHE(we have good code)
      5. green channel extraction (we have good code)
"""
import cv2
import numpy as np


class CLAHE:
    def __call__(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class GreenChannelExtraction:
      def __call__(self,image):
            return image[:,:,1]

class CropBlackBorder:
    def __init__(self, tol=7):
        self.tol = tol

    def __call__(self, img):

        if img.ndim == 2:

            mask = img > self.tol

            return img[
                np.ix_(
                    mask.any(axis=1),
                    mask.any(axis=0)
                )
            ]

        elif img.ndim == 3:

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY
            )

            mask = gray > self.tol

            if mask.sum() == 0:
                return img

            return img[
                np.ix_(
                    mask.any(axis=1),
                    mask.any(axis=0)
                )
            ]

        return img











# class CLAHE:
#       def __call__(self,image):
#             if(len(image.shape) == 3):
#                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             else:
#                  gray_image = image
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             return clahe.apply(gray_image)
      

