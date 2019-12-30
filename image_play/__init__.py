# import the necessary packages
#%matplotlib inline

import numpy as np
import requests
import cv2
import cv2.data
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from IPython.display import display

### PRETRAINED CLASSIFIERS
# Each classifer is an xml file, which, like JSON, is a "nested dictionary" style data structure.

### haarPath depends on the structure of colab instances, which could change
###  Worst case, #!git clone git://github.com/opencv/opencv.git for known path
haarPath = cv2.data.haarcascades
faceFrontDetector = haarPath + 'haarcascade_frontalface_default.xml'   # faces looking forward
faceProfileDetector = haarPath + 'haarcascade_profileface.xml'         # faces in profile
smileDetector = haarPath + 'haarcascade_smile.xml'                     # smiles
eyeDetector = haarPath + 'haarcascade_eye_tree_eyeglasses.xml'         # eyes
bodyDetector = haarPath + 'haarcascade_fullbody.xml'                   # full bodies
torsoDetector = haarPath + 'haarcascade_upperbody.xml'                 # upper bodies
catFaceDetector = haarPath + 'haarcascade_frontalcatface_extended.xml' # cat faces

### HELPER FUNCTIONS
   
# Retrieve kitten pictures of any size from placekitten.com
#   It's a Week 1 throwback!   
def placekittendotcomURL(w, h, color=True):
  """
  Construct URL for placekitten.com from parameters.
  """
  greyscale = "" if color else "g/"
  URL="http://placekitten.com/{}{}/{}".format( greyscale, w, h ) 
  return( URL )
  
# https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def url_to_image(url):
  """
  Given URL of an image on the web, retrieve it into a PIL Image object
  """
  headers = {'User-agent':'Mozilla/5.0'}
  resp = requests.get(url, headers=headers) 
  img = resp.content
  img = Image.open( BytesIO( img )).convert("RGBA")
  # return the image
  return img

# show image
def print_image(  img  ):
  """
  Dummy variable for notebook context, because print() doesn't print the visual 
  of an image object, but ending a cell with that object does, and display() does.
  
  A wrapper for display().
  """
  return( display( img ) )



# use classifer to both detect object and locate it in the image (with a rectangle of coordinates)
def locateObject(img, classifierPath='haarcascade_frontalface_default.xml'):
  """
  Takes PIL Image and the filename of an opencv classifier weights file.  
  Image is converted to np then to cv2 standard. detection is performed and 
    coordinates of foundobjects are returned.  
  The image is copied, annotated with coordinates of found objects, and returned.
  """
  #image = np.asarray( bytearray( img ), dtype="uint8")
  image = np.array(img, dtype=np.uint8)
  #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  object_cascade = cv2.CascadeClassifier(classifierPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  found_objects = object_cascade.detectMultiScale(gray, 1.3, 5)
  image_annotated = Image.fromarray(image.copy())
  draw = ImageDraw.Draw(image_annotated)
  for x,y,w,h in found_objects:
    draw.rectangle(((x, y), (x+w, y+h)), outline="blue")
  return( image_annotated)

# same as above but Boolean return value
def hasObject(img, classifierPath='haarcascade_frontalface_default.xml'):
  #image = np.asarray( bytearray( img ), dtype="uint8")
  image = np.array(img, dtype=np.uint8)
  #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  object_cascade = cv2.CascadeClassifier(classifierPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  found_objects = object_cascade.detectMultiScale(gray, 1.3, 5)
  return( bool(len(found_objects)))

# boolean cat detector
def hasCatFace(img):
  return( hasObject(img, classifierPath=catFaceDetector) )

# boolean face detector
def hasPersonFace(img):
  return( hasObject(img, classifierPath=faceFrontDetector) )

