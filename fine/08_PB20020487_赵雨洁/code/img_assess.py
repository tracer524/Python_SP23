# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
def SMD(img):#灰度方差函数
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
    	for y in range(1, shape[1]):
            out+=np.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=np.fabs(int(img[x,y]-int(img[x+1,y])))
    return out



# %%
def variance(img):#方差函数
   
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out


# %%
def entropy(img):#熵函数    
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*np.log(p[i]/count)/count
    return out

def smooth(image, sigma = 1.4, length = 5):
    """ Smooth the image
    Compute a gaussian filter with sigma = sigma and kernal_length = length.
    Each element in the kernal can be computed as below:
        G[i, j] = (1/(2*pi*sigma**2))*exp(-((i-k-1)**2 + (j-k-1)**2)/2*sigma**2)
    Then, use the gaussian filter to smooth the input image.

    Args:
        image: array of grey image
        sigma: the sigma of gaussian filter, default to be 1.4
        length: the kernal length, default to be 5

    Returns:
        the smoothed image
    """
    # Compute gaussian filter
    k = length // 2
    gaussian = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            gaussian[i, j] = np.exp(-((i-k) ** 2 + (j-k) ** 2) / (2 * sigma ** 2))
    gaussian /= 2 * np.pi * sigma ** 2
    # Batch Normalization
    gaussian = gaussian / np.sum(gaussian)

    # Use Gaussian Filter
    W, H = image.shape
    new_image = np.zeros([W - k * 2, H - k * 2])

    for i in range(W - 2 * k):
        for j in range(H - 2 * k):
            new_image[i, j] = np.sum(image[i:i+length, j:j+length] * gaussian)

    new_image = np.uint8(new_image)

    return new_image


def get_gradient_and_direction(image):
    """ Compute gradients and its direction
    Use Sobel filter to compute gradients and direction.
         -1 0 1        -1 -2 -1
    Gx = -2 0 2   Gy =  0  0  0
         -1 0 1         1  2  1

    Args:
        image: array of grey image

    Returns:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    W, H = image.shape
    gradients = np.zeros([W - 2, H - 2])
    direction = np.zeros([W - 2, H - 2])

    for i in range(W - 2):
        for j in range(H - 2):
            dx = np.sum(image[i:i+3, j:j+3] * Gx)
            dy = np.sum(image[i:i+3, j:j+3] * Gy)
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradients = np.uint8(gradients)

    return gradients, direction


def NMS(gradients, direction):
    """ Non-maxima suppression

    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0

    return nms


def double_threshold(nms, threshold1, threshold2):
    """ Double Threshold
    Use two thresholds to compute the edge.

    Args:
        nms: the input image
        threshold1: the low threshold
        threshold2: the high threshold

    Returns:
        The binary image.
    """
    visited = np.zeros_like(nms)
    output_image = nms.copy()
    W, H = output_image.shape

    def dfs(i, j):
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > threshold1:
            output_image[i, j] = 255
            dfs(i-1, j-1)
            dfs(i-1, j)
            dfs(i-1, j+1)
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i+1, j-1)
            dfs(i+1, j)
            dfs(i+1, j+1)
        else:
            output_image[i, j] = 0


    for w in range(W):
        for h in range(H):
            if visited[w, h] == 1:
                continue
            if output_image[w, h] >= threshold2:
                dfs(w, h)
            elif output_image[w, h] <= threshold1:
                output_image[w, h] = 0
                visited[w, h] = 1

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 0:
                output_image[w, h] = 0

    return output_image
# %%

def  GradientEdgeFun(img):
    h=img.shape[0]
    w=img.shape[1]
    imShape=img.shape
    dim=len(imShape)
    
    eim=np.zeros(img.shape)
    for i in range(img.shape[0]-1):
             for j in range(img.shape[1]-1):
                 eim[i,j]=abs(img[i,j+1]-img[i,j])+abs(img[i+1,j]-img[i,j])
    plt.imshow(eim)
    plt.show()

# %%
def canny(img):
    for x in range(13,17,2):
        for y in range(60,80,5):
            blurred = cv2.GaussianBlur(img,(x,x),0)
            gaussImg = cv2.Canny(blurred, 10, y)
            cv2.imwrite('sag'+str(x)+'_'+str(y)+".png",gaussImg)

# %%

if __name__ == "__main__":
    img_path = "canny_img/T1_axis.png"
    img = cv2.imread(img_path,0)
    #获取图片的宽和高
    width,height = img.shape[:2][::-1]
    print("img_reisze shape:{}".format(np.shape(img)))
    print(variance(img))
    print(entropy(img))
    GradientEdgeFun(img)
    canny(img)


