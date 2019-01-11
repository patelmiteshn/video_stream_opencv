import os
import cv2 
import Equirec2Perspec as E2P 

if __name__ == '__main__':
    equ = E2P.Equirectangular('../equi/1_equi.JPG')    # Load equirectangular image
    
    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    #
    img = equ.GetPerspective(80, -90, 0, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
    cv2.imwrite('perspective1280x960_80_-90.jpg', img)