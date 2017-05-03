
# coding: utf-8

# In[1]:

import os
from collections import Counter
from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# In[2]:

def horizontal_project(im, threshold=0):
    projection = []
    width = im.shape[1]
    for j in range(width):
        projection.append(Counter(im[:,j])[threshold]) #black pot number
        
    return projection


# In[3]:

def compute_cutline(projection, only_one=False):
    "x0, x1 all included"
    state='start'
    break_pos = []
    x0=None
    x1=None
    zero_count=0
        
    for i,n in enumerate(projection):
        
        if state == 'start':
            if n == 0:
                state = 'zero'
            else:
                state = 'none-zero'
                x0 = i
                
        elif state == 'zero':
            if n != 0:
                state = 'none-zero'
                x0 = i
                
        elif state == 'none-zero':
            if n == 0:
                state = 'zero'
                x1 = i
                break_pos.append((x0, x1))

               
    if only_one and len(break_pos)>1:
        break_pos = [(break_pos[0][0], break_pos[-1][1])]
        
    return break_pos


# In[4]:

def vertical_project(im, threshold=0):
    projection = []
    height = im.shape[0]
    for i in range(height):
        projection.append(Counter(im[i,:])[threshold]) #black pot number
        
    return projection


# In[5]:

def format_screen(im):
    img = Image.fromarray(im)
    img_enh=ImageEnhance.Sharpness(img).enhance(3)
    img_enh=img_enh.resize((70, 20), Image.ANTIALIAS)
    im = np.asarray(img_enh, dtype=np.float32)

    return im

def load_gray_image(im_path):
    im = cv2.imread(im_path)
    if im.shape != (20, 70):
        im = format_screen(im)
    im = im[1:-1, 1:-1]
    im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    return im_gray


# In[7]:

def _isvalid_point(x,y,h,w):
    if x>=0 and x<h and y>=0 and y<w:
        return True
    else:
        return False
    

def clear_noise(im, threshold=140): #threshold for blank point!!
    """im is instance of numpy.array, gray image"""
    h, w = im.shape

    isvalid = partial(_isvalid_point, w=w, h=h)
    noise_point_set=[]
    for x in range(h):
        for y in range(w):
            blank_count = 0
            valid_count = 0
            if isvalid(x-1,y):valid_count+=1
            if isvalid(x+1,y):valid_count+=1
            if isvalid(x,y-1):valid_count+=1
            if isvalid(x,y+1):valid_count+=1
                
            if isvalid(x-1,y-1):valid_count+=1
            if isvalid(x+1,y-1):valid_count+=1
            if isvalid(x-1,y+1):valid_count+=1
            if isvalid(x+1,y+1):valid_count+=1
                
            if isvalid(x-1,y) and im[x-1,y] >= threshold: # up
                blank_count += 1
                
            if isvalid(x+1,y) and im[x+1,y] >= threshold: # down
                blank_count += 1
                
            if isvalid(x,y-1) and im[x,y-1] >= threshold: # left
                blank_count += 1
                
            if isvalid(x,y+1) and im[x,y+1] >= threshold: # right
                blank_count += 1
            
            if isvalid(x-1,y-1) and im[x-1,y-1]>=threshold: # uppper left 
                blank_count += 1
                
            if isvalid(x+1, y-1) and im[x+1, y-1] >=threshold: # bottom left
                blank_count += 1
                
            if isvalid(x-1, y+1) and im[x-1,y+1]>=threshold: # upper right
                blank_count += 1
                
            if isvalid(x+1,y+1) and im[x+1, y+1]>=threshold: # bottom right
                blank_count += 1
                
            
            if blank_count == valid_count:
                #print(x,y, blank_count)
                noise_point_set.append((x,y))
                
    for x,y in noise_point_set:
        im[x,y] = threshold

    return im


# In[11]:

def _split_letters(im_b):
    projection_h = horizontal_project(im_b)
    cutlines_h = compute_cutline(projection_h)
    image_split_v = [im_b[:,line[0]:line[1]] for line in cutlines_h]
    
    letters = []
    for i, each_image_split_v in enumerate(image_split_v):
        projection_v=vertical_project(each_image_split_v)
        cutlines_v = compute_cutline(projection_v, only_one=True)
        #print(cutlines_v)
        line = cutlines_v[0]
        letters.append(image_split_v[i][line[0]:line[1],:])
        
    return letters

def format_letter(im, out_height=32, out_width=32):
    
    offset_x = int(abs(out_height - im.shape[0]) / 2)
    offset_y = int(abs(out_width - im.shape[1]) / 2)

    im_height, im_width = im.shape
    
    out = np.ones((out_height, out_width)) * 255
    out[offset_x: offset_x+im_height, offset_y: offset_y+im_width] = im
    
    return out
    
    
def split_letters(im_path):
    """
    
    :param im_path: 
    :return: list<nump.ndarray> 
    """
    im_gray = load_gray_image(im_path)
    _, im_b = cv2.threshold(im_gray, 140, 255, cv2.THRESH_BINARY)
    im_b = clear_noise(im_b, threshold=255)
    letters = [format_letter(letter) for letter in _split_letters(im_b)]
    #print(letters)
    return letters


# In[8]:

def split_test(input_dir='captchas', out_dir='dataset'):
    for fn in os.listdir(input_dir):
        bare_fn, ext = os.path.splitext(fn)
        im_path = os.path.join(input_dir, fn)
        try:
            letters = split_letters(im_path)
        except Exception as ex:
            print(ex, im_path)
            continue
        for i, letter in enumerate(letters):
            fn2 = bare_fn + '_%d' % i + ext

            cv2.imwrite(os.path.join(out_dir, fn2), letter)
            print('splited %s' % im_path)
                
def cli():
    import sys
    im_path = sys.argv[1]
    letters = split_letters(im_path)
    bare_fn, ext = os.path.splitext(im_path)
    for i,letter in enumerate(letters):
        fn2 = bare_fn + '_%d'%i + ext
        cv2.imwrite(fn2, letter)

if __name__ == '__main__':
    cli()

