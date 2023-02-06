from numpy import ndarray, array, cross
import numpy as np
from numpy.linalg import solve, norm
import time




def __detect_center_old(ray_list, d=1000):
    ray_count = len(ray_list)
    # print(ray_count)
    """
        ray_list: [[two points on ray0], [two points on ray1], ...]
    """
    centers = []
    # doğrulardaki en yakın noktaların birbirinden uzak olması durumunda uygulanan ayıklama
    for i in range(0,len(ray_list)):
        for j in range(i+1,len(ray_list)):
            center=ray_center(ray_list[i][0],ray_list[i][1],ray_list[j][0],ray_list[j][1])
            if norm(center[1]-center[2])<d:
                centers.append(center[0])

    if len(centers) == 0:
        return None

    # ikinci ayıklamadan sonra verilerin depolanacağı liste
    real_centers=[]

    #doğrulardaki en yakın noktaların birbirine yakın ama diğer her noktaya uzak olması durumunda
    # uygulanan ayıklama, mantığı least square mantığına benziyor
    for i in range(0,len(centers)):
        score=0
        for j in range(0,len(centers)):
            score+=norm(centers[i]-centers[j])
        score /= len(centers)
        if score<d:
            real_centers.append(centers[i])
            
            
    # print(real_centers)
    if len(real_centers) ==0:
        return None

    one_true_center=real_centers[0]

    for i in range(1,len(real_centers)):
        one_true_center+=real_centers[i]
        
    one_true_center=one_true_center/len(real_centers)

    return one_true_center

def ray_center(ray_0_point_0,ray_0_point_1,ray_1_point_0,ray_1_point_1):

    #2 noktadan geçen doğru için gereken noktalar
       
    
    # bu noktalardan geçen vektör
    ray_0_direction = (ray_0_point_1 - ray_0_point_0) / norm(ray_0_point_1 - ray_0_point_0)
    ray_1_direction = (ray_1_point_1 - ray_1_point_0) / norm(ray_1_point_1 - ray_1_point_0)
    # print(norm(ray_1_point_1 - ray_1_point_0, axis=1))
    #matematik
    
    UC = cross(ray_1_direction, ray_0_direction)
    UC /= norm(UC)
    
    
    RHS = ray_1_point_0 - ray_0_point_0
    LHS = array([ray_0_direction, -ray_1_direction, UC]).T 
    
    t=solve(LHS, RHS)

    # ilk doğrudan geçen nokta
    
    D1 = ray_0_point_0 + t[0]*ray_0_direction
    
    # ikinci doğrudan geçen nokta
    
    D2 = ray_1_point_0 + t[1]*ray_1_direction
    
    # merkez noktası
    
    center= (D1+D2)/2
    
    # print(center)
    
    return center,D1,D2



def multi_ray_center(ray_0_point_0,ray_0_direction,other_rays_point_0,other_rays_direction):
    ray_count = other_rays_point_0.shape[1]
    #2 noktadan geçen doğru için gereken noktalar
       
    # #matematik
    
    UC = cross(other_rays_direction, ray_0_direction, axis=0)
    # UC /= norm(UC, axis=0)
    
    
    RHS = other_rays_point_0 - ray_0_point_0
    a = np.broadcast_to(ray_0_direction, other_rays_direction.shape)
    b=-other_rays_direction
    c=UC
    LHS = np.stack([a, b, c], axis=0).T 
    RHS = RHS.T.reshape(other_rays_direction.shape[1],3,1)
    t=solve(LHS, RHS)
    
    # ilk doğrudan geçen nokta
    
    D1 = ray_0_point_0 + np.broadcast_to(ray_0_direction, (3,ray_count)) * t[:,0,:].T

    # ikinci doğrudan geçen nokta
    
    D2 = other_rays_point_0 + other_rays_direction * t[:,1,:].T
    
    # merkez noktası
    
    center= (D1+D2)/2
    
    
    return center,D1,D2   

def detect_center(ray_list, d=1000):
    ray_count = len(ray_list)
    # print(ray_count)
    """
        ray_list: [[two points on ray0], [two points on ray1], ...]
    """
    centers = None
    ray_points_0 = np.hstack([ray[0].reshape(3,1) for ray in ray_list])
    ray_directions= np.hstack([ray[1].reshape(3,1) for ray in ray_list])
    ray_directions = ray_directions / norm(ray_directions, axis=0)
    # doğrulardaki en yakın noktaların birbirinden uzak olması durumunda uygulanan ayıklama
    for i in range(0,len(ray_list)):
        ray_0_point_0 = ray_points_0[:,i:i+1]
        ray_0_direction = ray_directions[:,i:i+1]
        other_rays_point_0 = ray_points_0[:,i+1:]
        other_rays_direction = ray_directions[:,i+1:]
        res = multi_ray_center(ray_0_point_0, ray_0_direction, other_rays_point_0, other_rays_direction)
        centers_interim = res[0]
        D1 = res[1]
        D2 = res[2]
        # print(centers_interim)
        centers_interim = centers_interim.T[np.where(norm(D1-D2,axis=0)<d)].T
        if centers is None:
            centers = centers_interim
        else:
            centers = np.hstack((centers, centers_interim))

    if centers.shape[1] == 0:
        return None

    # ikinci ayıklamadan sonra verilerin depolanacağı liste
    real_centers=None

    #doğrulardaki en yakın noktaların birbirine yakın ama diğer her noktaya uzak olması durumunda
    # uygulanan ayıklama, mantığı least square mantığına benziyor
    center_count = centers.shape[1]
    a = centers.T.reshape(1,center_count,3)
    b = centers.T.reshape(center_count,1,3)
    score = np.sum(norm(a - b, axis=2), axis=0) / center_count
    real_centers = centers.T[np.where(score<d)].T
            
            
    # print(real_centers)
    if real_centers.shape[1] ==0:
        return None

    return np.average(real_centers, axis=1)

class RayCenterCalculator(object):
    def __init__(self,d:float):
        self.d = d
        self._real_centers=None
        self.ray_origins=None
        self.ray_directions=None
        self._centers=None
    
    def add_ray(self, ray:np.ndarray):
        ray_origin = ray[0]
        ray_direction = ray[1] / norm(ray[1])
        if self.ray_origins is None:
            self.ray_origins = ray_origin
            self.ray_directions = ray_direction
            return None
        
        res = multi_ray_center(ray_origin, ray_direction, self.ray_origins, self.ray_directions)
        self.ray_origins = np.hstack((self.ray_origins, ray_origin))
        self.ray_directions = np.hstack((self.ray_directions, ray_direction))

        centers_interim = res[0]
        D1 = res[1]
        D2 = res[2]

        centers_interim = centers_interim.T[np.where(norm(D1-D2,axis=0)<self.d)].T
        if self._centers is None:
            self._centers = centers_interim
        else:
            self._centers = np.hstack((self._centers, centers_interim))

        if self._centers.shape[1] == 0:
            return None

        # a = centers_interim.T.reshape(1,-1,3)
        # b = np.hstack((self._centers, centers_interim)).T.reshape(-1,1,3)
        # score = np.average(norm(a - b, axis=2), axis=0)
        # real_centers_interim = centers_interim.T[np.where(score<d)].T

        # if self._real_centers is None:
        #     self._real_centers = real_centers_interim
        # else:
        #     self._real_centers = np.hstack((self._real_centers, real_centers_interim))
        
        # if self._real_centers.shape[1] == 0:
        #     return None
        
        # return np.average(self._real_centers, axis=1) 
        return np.average(self._centers, axis=1) 

        
def oop_test(rays, d):
    calculator = RayCenterCalculator(d)
    result = None
    for ray in rays:
        result = calculator.add_ray(ray)
    # result = calculator._test(rays, d)
    return result


        
if __name__=="__main__":
    # XA0 = array([[1, 0, 0]]).T
    # XA1 = array([[1, 2, 3]]).T
    # XB0 = array([[1, 0, 0]]).T
    # XB1 = array([[1, 2, 4]]).T
    # XC0 = array([[1, 0, 0]]).T
    # XC1 = array([[1, 2, 5]]).T
    
    # center = ray_center(XA0.reshape(3),XA1.reshape(3),XB0.reshape(3),XB1.reshape(3))
    # print(center)

    # center=multi_ray_center(XA0,XA1,np.hstack((XB0,XC0)),np.hstack((XB1,XC1)))
    # print(center)

    # N=20
    # ray_0_point_0 = np.random.rand(3,1)
    # ray_0_point_1 = np.random.rand(3,1)
    # other_rays_point_0 = np.random.rand(3,N)
    # other_rays_point_1 = np.random.rand(3,N)

    # centers = multi_ray_center(ray_0_point_0, ray_0_point_1, other_rays_point_0, other_rays_point_1)[0]
    # # print(centers)
    # for i in range(N):
    #     center = ray_center(ray_0_point_0.reshape(3), ray_0_point_1.reshape(3), other_rays_point_0[:,i].reshape(3), other_rays_point_1[:,i].reshape(3))[0]
    #     print(np.allclose(center, centers[:,i]))

    N=100
    d=1000
    rays = [[np.random.rand(3,1) for i in range(2)] for j in range(N)]

    # start = time.perf_counter()
    # print(__detect_center_old(rays, d))
    # print(time.perf_counter()-start)

    start = time.perf_counter()
    print(detect_center(rays, d))
    print(time.perf_counter()-start)
    
    start = time.perf_counter()
    print(oop_test(rays, d))
    print(time.perf_counter()-start)
        
        
















    
