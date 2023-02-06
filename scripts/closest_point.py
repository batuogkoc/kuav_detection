from numpy import array, cross
from numpy.linalg import solve, norm




# # verilerin geldiği liste
# liste= [[]]

# # verileri temizlemek için kullanılacak uzaklık threshold
# d= 1000

# ayıklama olmadan depolanan centerlar
centers=[]


def ray_center(XA0,XA1,XB0,XB1):

    #2 noktadan geçen doğru için gereken noktalar
    
    
    
    # bu noktalardan geçen vektör
    
    UA = (XA1 - XA0) / norm(XA1 - XA0)
    UB = (XB1 - XB0) / norm(XB1 - XB0)
    
    #matematik
    
    UC = cross(UB, UA)
    UC /= norm(UC)
    
    
    RHS = XB0 - XA0
    LHS = array([UA, -UB, UC]).T
    t=solve(LHS, RHS)
    
    # ilk doğrudan geçen nokta
    
    D1 = XA0 + t[0]*UA
    
    # ikinci doğrudan geçen nokta
    
    D2 = XB0 + t[1]*UB
    
    # merkez noktası
    
    center= (D1+D2)/2
    
    # print(center)
    
    return center,D1,D2

def detect_center(ray_list, d=1000):
    ray_count = len(ray_list)
    print(ray_count)
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
            
        
if __name__=="__main__":
    XA0 = array([1, 0, 0])
    XA1 = array([1, 2, 3])
    XB0 = array([1, 0, 0])
    XB1 = array([1, 2, 4])

    center = ray_center(XA0,XA1,XB0,XB1)
    print(center[0])

    center=detect_center([(XA0,XA1),(XB0,XB1)])
    print(center)
        
        
        
















    
