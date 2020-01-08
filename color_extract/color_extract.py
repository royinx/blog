import cv2
# from skimage.color import rgb2hsv, hsv2rgb
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.cluster import KMeans , DBSCAN
import numpy as np 
import matplotlib.pyplot as plt

class ColorFilter(object):
    def __init__(self):
        pass
    
    def color_filter(self, image, colors, std = 20): # Inference 
        color_percent = []
        for color in colors:
            color_pixel_array = np.where(np.all([np.logical_and(image[:,:,0]>=color[0]-std, image[:,:,0]<=color[0]+std),
                                                 np.logical_and(image[:,:,1]>=color[1]-std, image[:,:,1]<=color[1]+std),
                                                 np.logical_and(image[:,:,2]>=color[2]-std, image[:,:,2]<=color[2]+std)],
                                                 axis=0))
            color_percent.append(len(image[color_pixel_array])/(image.size/3)*100)
        return color_percent

    def color_match(self, img, colors: np.array , target_percent: np.array ,color_fmt ,color_threshold = 10 ): # image and pixel color matching , color_threshold: percent_std_dev, color_range: RGB +- 15
        if color_fmt=='hsv':
            color_range = 0.2
        elif color_fmt=='rgb':
            color_range = 20
        else:
            raise('Invalid color format')

        color_percent = self.color_filter(img, colors, std = color_range)
        bl = np.logical_and(color_percent >= target_percent - color_threshold , color_percent <= target_percent + color_threshold)

        return np.all(bl) # match or not

    def dominantColors_Kmeans(self, image, k=5):
        #reshaping to a list of pixels
        img = image.reshape((image.shape[0] * image.shape[1], 3))
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(img)

        #the cluster centers are our dominant colors.
        colors = kmeans.cluster_centers_
        
        #save labels
        labels = kmeans.labels_
        
        #returning after converting to integer from float
        return colors.astype(int)


    # Caution: size of image < 250*250,3 as possible
    def dominantColors_DBSCAN(self, image,color_fmt, plot, eps=5, min_sample = 100):
        #reshaping to a list of pixels
        img = image.reshape((image.shape[0] * image.shape[1], 3))

        dbs = DBSCAN(eps,min_sample)

        idx = dbs.fit_predict(img)
        no_class = len(set(idx))

        # print("dtype:",image.dtype)
        print("============ no_class: {} ============".format(no_class))
        if color_fmt == 'hsv':
            colors = np.array([np.average(img[idx==class_],axis = 0) for class_ in range(no_class)])#,dtype=np.uint8)
            percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(no_class)])#,dtype=int)
            def plot_HSV(img,idx):
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
                import matplotlib.pyplot as plt

                fig = plt.figure()
                fig.set_size_inches(20, 15)
                ax = fig.add_subplot(111, projection='3d')

                xs = img[:,0]
                ys = img[:,1]
                zs = img[:,2]
                ax.scatter(xs, ys, zs, marker='o', c=idx)

                ax.set_xlabel('H Label')
                ax.set_ylabel('S Label')
                ax.set_zlabel('V Label')
                plt.show()
            if plot:
                plot_HSV(img,idx)
            
        elif color_fmt == 'rgb':
            colors = np.array([np.average(img[idx==class_],axis = 0) for class_ in range(no_class)],dtype=np.uint8)
            percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(no_class)],dtype=int)
            def plot_RGB(img,idx):
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
                import matplotlib.pyplot as plt

                fig = plt.figure()
                fig.set_size_inches(20, 15)
                ax = fig.add_subplot(111, projection='3d')

                xs = img[:,0]
                ys = img[:,1]
                zs = img[:,2]
                ax.scatter(xs, ys, zs, marker='o', c=idx)

                ax.set_xlabel('R Label')
                ax.set_ylabel('G Label')
                ax.set_zlabel('B Label')

                plt.show()
            if plot:
                plot_RGB(img,idx)

        else:
            raise('Invalid color format')
        return colors, percent

    def resize(self, image,size=224):
        scale = image.shape[1]/image.shape[0]  # W/H
        inp = cv2.resize(image,(int(size*scale),size), interpolation = cv2.INTER_LINEAR) #(resize shape = (w,h) )
        inp = inp.astype(np.uint8)
        return inp

    def extraction(self, image, color_fmt = 'hsv',plot = False):

        if color_fmt == 'hsv':
            eps = 0.05
            min_sample = 50
        elif color_fmt == 'rgb':
            eps = 3.5
            min_sample = 20

        color_list, percent = self.dominantColors_DBSCAN(image,color_fmt, plot,eps, min_sample)
        return color_list, percent

def extract_color(rgb_img:np.array, color_fmt = 'hsv'):
    color_filter = ColorFilter()
    rgb_img = color_filter.resize(rgb_img,100)

    # print(img[0,0])
    if color_fmt == 'hsv':
        hsv_img = rgb_to_hsv(rgb_img/255)
    color_list, percent = color_filter.extraction(hsv_img,color_fmt,plot = False)

    # convert color_list into RGB 
    hsv_color_list = color_list
    if color_fmt == 'hsv':
        rgb = []
        for idx,color in enumerate(color_list):
            rgb.append((hsv_to_rgb(np.tile(color.reshape(1,3),[1,1,1]))*255).astype(np.uint8))
        color_list = np.array(rgb).squeeze()

    print(color_list,percent)
    
    # bl = color_filter.color_match(rgb_img, color_list, percent)
    bl = color_filter.color_match(hsv_img, hsv_color_list, percent, color_fmt)
    
    return(bl)

if __name__ == '__main__':
    file_name = 'jacket.jpg'
    img = cv2.imread(file_name)
    rgb_img = img[:,:,[2,1,0]]

    # plt.imshow(rgb_img)
    # plt.show()
    extract_color(rgb_img,color_fmt = 'hsv')

