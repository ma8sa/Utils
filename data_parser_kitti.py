import cv2
import numpy as np
import os
import sys 
import random


def gt_bbox_check(bbox,p):
    
    pt = list(map(int,p[0]))
    
    for b in bbox:
        #TODO check for x,y order : i think it works 
        if (b[0] < pt[0] and b[1] < pt[1] and pt[0] < b[2] and pt[1] < b[3]) : 
            return True 
    return False

def bbox_check(bbox,p):
    
    pt = list(map(int,p[0]))
     
    for b in bbox:
        #TODO check for x,y order : i think it works 
        if (b[0] < pt[0] and b[1] < pt[1] and pt[0] < b[2] and pt[1] < b[3]) : 
            return True 
    return False

def random_color():
    rgbl=[255,0,150,50,170,200,120]
    random.shuffle(rgbl)
    return tuple(rgbl[:3])

def draw_bbox(img,bbox,f_info):
   
    for i,b in enumerate(bbox):
        clr = random_color()
        cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),clr,2)
        cv2.putText(img,"S: " +f_info[i][1] ,(b[0],b[1]),0,0.9,clr)
       
    return img 
    
    
def parse_bbox(seq_no=0):
    pre = './label_02/'
    stuff = np.loadtxt(pre + str(seq_no).zfill(4) + '.txt',dtype=np.str,delimiter=' ')
    print(stuff[-1][0])
    frames = int(stuff[-1][0])
    #info = [ [ i if cur_f==int(i[0]) else cur_f += 1 for i in stuff ] ]
    tmp   = []
    info  = []
    cur_f = 0

    for i in stuff:

        if cur_f == int(i[0]):
           tmp.append(i)
        else:
            info.append(tmp)
            cur_f += 1
            tmp = []

    info.append(tmp)
     
    return info
   
def draw_points(img,X,Y,gt):

    cls = [ (0,255,0), (0,0,255) , ( 255,0,0) ]

    for i,x in enumerate(X):
        cv2.circle(img,(int(X[i]),int(Y[i])),2,cls[int(gt[i])],-1)

    return img
 
    # arrage the file accordiung to frames
def read_bbox(info,seq_no=0,st_frame=0,window=20):
    
    #info = [ ]
    
    image = './image_02/' + str(seq_no).zfill(4) + '/' + str(st_frame).zfill(6) + '.png'
    img  = cv2.imread(image)
    print(image)
    input()
    
    # just go to start frame and get all bbox within the list
    gt_list = [ "Van" , "Cyclist"]
    #gt_list = [ "Van","Car" ]
    f_list = [ "Van" , "Cyclist", "Car","Truck","Tram","Pedestrian"]
    f_bbox = [] 
    f_info = [] 
    gt_bbox = [] 
    gt_info = [] 

    for i in info[st_frame]:

          if i[2] in f_list:
             
             f_bbox.append(i[6:10])
             f_info.append(i[:5])

          if i[2] in gt_list:
             gt_bbox.append(i[6:10])
             gt_info.append(i[1])
      
    gt_bbox = [ [ int(float(x)) for x in i] for i in gt_bbox ]
    f_bbox = [ [ int(float(x)) for x in i] for i in f_bbox ]

    img = draw_bbox(img,f_bbox,f_info)
    cv2.imshow('image',img)
    cv2.waitKey()
    #input()
    cv2.destroyAllWindows()
    print(" total bbox in scene :{}".format(len(f_bbox) ) ) 
    mov = input("no. of moving bboxes: \n")
    inds_mov = []
    st_bbox = []
    gt_bbox = []
    
       
    for i in range(int(mov)):
          inds_mov.append((input()))
    print(" bbox st frame {} , end Frame {}".format(st_frame-window,st_frame)) 
    for f in info[st_frame-window:st_frame+1]:
          tmp = []
          
          for i in f:
              if i[1] in inds_mov:
                 tmp.append(i[6:10])

          tmp = [ [ int(float(x)) for x in i] for i in tmp ]
          gt_bbox.append(tmp)
    
    print(inds_mov)
    input()

    #for i in range(len(f_bbox)):

      #  if i in inds_mov:
           #gt_bbox.append(f_bbox[i])
       # else: 
        #   st_bbox.append(f_bbox[i])
        

    return f_bbox,gt_bbox,st_bbox


def make_ground_truth(seq_no,st_frame,window,tracked_points,bbox,f_bbox,x,y,img):
    # cehck wheter the points fall in the the moving bbox
    # read all bbox

    #TODO list of moving bbox in the given window with just the st frame as refence check all the window frames for checking
    inds = []
    for w in range(window): 
         inds.append( [ 1 if ( bbox_check(bbox[w],xp)) else 0  for i,xp in enumerate(tracked_points[w]) ] )

    inds = np.asarray(inds)
    inds = np.mean(inds, axis=0)
    # TODO set threshold here
    inds = inds >= 0.9
    
    l,_,_ = (tracked_points[0].shape)
    gt = np.ones(l) 
    gt[inds] = 0.0 
    # TODO make an iterwative bbox machanism
 
    img = draw_points(img,x[0],y[0],gt)
    clone = img.copy()
    rect_pts = []
    neg_pts = []
    win_name = " points"
    print(" SELECT POINTS ")
    def select_points(event, px, py, flags, param):

        nonlocal rect_pts,neg_pts,gt,clone,x,y,img
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(px, py)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((px, py))
            
            print("drawing")
            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)
            tmp_bbox = [list(rect_pts[0] + rect_pts[1] )]
            
            gt = [ 0 if( bbox_check(tmp_bbox,xp) ) else gt[i] for i , xp in enumerate(tracked_points[0]) ]
            img = draw_points(clone,x[0],y[0],gt)

        if event == cv2.EVENT_RBUTTONDOWN:
            neg_pts = [(px, py)]

        if event == cv2.EVENT_RBUTTONUP:
            neg_pts.append((px, py))
            
            print("drawing")
            # draw a rectangle around the region of interest
            cv2.rectangle(clone, neg_pts[0], neg_pts[1], (0,0, 255), 2)
            cv2.imshow(win_name, clone)
            tmp_bbox = [list(neg_pts[0] + neg_pts[1] )]
            
            gt = [ 1 if( bbox_check(tmp_bbox,xp) ) else gt[i] for i , xp in enumerate(tracked_points[0]) ]
            img = draw_points(clone,x[0],y[0],gt)

    cv2.namedWindow(win_name)   
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = img.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break
    print("drawing done , press any key to procede")
    # close the open windows
    cv2.destroyWindow(win_name)

    return gt


def make_features(seq_no,st_frame,tracked_points,bbox):
    #TODO list of probale moving boxes ( load bbox with apporaeite labels)
    # check which all points fit in the bbox
     
    inds = [ i for i,x in enumerate(tracked_points[0]) if bbox_check(bbox,x) ]
    l,_,_ = (tracked_points[0].shape)
    ft = np.ones(l) 
    ft[inds] = 0.5 
    return ft
   
     
def save_tracks(tracked_points,info,seq=0,st_frame=0,set_no=0,window=20,path='./'):
    
    x = np.asarray([ i[:,0,0] for i in tracked_points ])
    y = np.asarray([ i[:,0,1] for i in tracked_points ])
    
    path = path + 'dataset_' + str(window).zfill(3) + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    f_bbox,gt_bbox,st_bbox = read_bbox(info,seq,st_frame,window) 
    f = make_features(seq,st_frame,tracked_points,f_bbox)

    color = np.random.randint(0,255,(21000,3)) 

    input("tracks_preview\n")
    for xx in range(3,window+1):
        image = './image_02/' + str(seq).zfill(4) + '/' + str(st_frame-window+xx).zfill(6) + '.png'
        print(image)
        img  = cv2.imread(image)
        img = draw_points(img,x[xx],y[xx],f)

        for ii,_ in enumerate(x[xx]):
            img = cv2.line(img, (x[xx,ii],y[xx,ii]),(x[xx-1,ii],y[xx-1,ii]), color[ii].tolist(), 2)
            
    #xx = np.loadtxt('test.txt',dtype=np.float,delimiter=' ')
        cv2.imshow('image',img)
        cv2.imwrite(path + '/tracks_' + str(seq).zfill(4) + '_set_' + str(set_no).zfill(2) +"__"+str(xx) + '.png',img)
        cv2.waitKey(2500)

    cv2.waitKey(3500)
    cv2.destroyAllWindows()
    image = './image_02/' + str(seq).zfill(4) + '/' + str(st_frame-window).zfill(6) + '.png'
    print(" gt image : {}".format(image))
    img  = cv2.imread(image)

    gt = make_ground_truth(seq,st_frame,window,tracked_points,gt_bbox,st_bbox,x,y,img)
    
    f = np.array([ 0 if gt[i]== 0 else x for i,x in enumerate(f) ] )

    np.savetxt(path + '/X_'+str(seq).zfill(4) + '_set_'+str(set_no).zfill(2) + '.txt',x,delimiter=' ') 
    np.savetxt(path + '/Y_' + str(seq).zfill(4)+'_set_'+ str(set_no).zfill(2) + '.txt',y,delimiter=' ') 
    np.savetxt(path + '/gt_' + str(seq).zfill(4)+'_set_'+ str(set_no).zfill(2) + '.txt',gt,delimiter=' ') 
    np.savetxt(path + '/f_' + str(seq).zfill(4)+'_set_'+ str(set_no).zfill(2) + '.txt',f,delimiter=' ') 
    

    cv2.imwrite(path + '/preview_' + str(seq).zfill(4) + '_set_' + str(set_no).zfill(2) + '.png',img)
    
     
    cv2.imwrite(path + '/GT_image_' + str(seq).zfill(4) + '_set_' + str(set_no).zfill(2) + '.png',img)
    set_no += 1

    return set_no

def detect_keypoints(img):
   
   gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   #surf = cv2.SURF(400)
   surf = cv2.xfeatures2d.SIFT_create(2000)
   test_mask = np.zeros(gray.shape, np.uint8)
   print(test_mask.shape)
   input()
   #test_mask[200:390,100:1800] = 1
   #test_mask = cv3.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
 
   #kp,_ = surf.detectAndCompute(gray,mask = test_mask)
   kp,_ = surf.detectAndCompute(gray,mask = None)
   #kp = np.float32(kp)
   pts = np.asarray([kp[idx].pt for idx in range(len(kp))]).reshape(-1, 1, 2)
   pts = np.float32(pts)
   corners = cv2.goodFeaturesToTrack(gray,2500,0.05,5,mask = test_mask)
   corners = np.float32(corners)
   print(type(corners))
   print(type(pts))
   print(corners.size)
   print(pts.size)
   input()

   return pts 
   
def track_points(img1,img2,kp0,f=0,stt=0):
  
    color = np.random.randint(0,255,(21000,3)) 
    old_gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    new_gray= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.8))

    mask = np.zeros_like(img2)
    #kp0 = np.asarray(kp0[:,:])
    kp1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, kp0, None, **lk_params)
    # Select good points
    good_new = kp1[st==1]
    good_old = kp0[st==1]
    # matching backwards
    p_old, st2, err = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, kp1, None, **lk_params)

    good_new = kp1[st2==1]
    good_old = p_old[st2==1]
    # as seen in official opencv example
    diff = abs(p_old - kp0).reshape(-1,2).max(-1)
    print(diff.shape)

    # track quality threshold
    good = diff < 3.
    st = np.array( [s if good[i] and (st2[i] == 1) else 0 for i,s in enumerate(st) ])

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        img2 = cv2.circle(img2,(a,b),5,color[i].tolist(),-1)
    
    img = cv2.add(img2,mask)
    cv2.imwrite("track_" +str(stt).zfill(6) + ".png",img)
    return kp1,st 


def run(start_frame=0,window_length=15,root_dir='./image_02/',seq_no=0,save_path='./'):
    
    # start frame
    # window
    # intiazlize track arrays
    # find features points
    # track points
    # append to track array
    # update track array usings st and change track array
    # update current frame , next frame , kp

    pre = root_dir + str(seq_no).zfill(4) + '/'
    fr = [ f for f in os.listdir(pre) if f.endswith('.png') ]
    window = len(fr)
        
    info = parse_bbox(seq_no)
    #window = 30
    print(" seq_no : {} , No of frames {} ".format(seq_no,window))

    st_frame = start_frame
    nxt_frame = start_frame + 1
    tracked_points = []

    first_image = pre + str(st_frame).zfill(6) + '.png' 
    first_img = cv2.imread(first_image)

    key_points = detect_keypoints(first_img)
    print(key_points.shape)
    tracked_points.append(key_points)
    
    set_no = 0
    for st_x in range(st_frame,window):
      for i in range(window_length+1):
        
        current_image = pre + str(st_frame+i).zfill(6) + '.png' 
        nxt_image = pre + str(st_frame+i+1).zfill(6) + '.png' 
        print(nxt_image)
       
        img = cv2.imread(current_image)
        img2 = cv2.imread(nxt_image)
       
        kp0 = tracked_points[-1]
        kp1,st = track_points(img,img2,kp0,0,i)
         
        tracked_points.append(kp1) 
        # update valid points in tracked points
        tracked_points = [ t[st==1].reshape(-1,1,2) for t in tracked_points ]
        
        print(" index {}, window {}".format(i,window_length) )
        if i>=window_length and i > 0:
           # TODO parse points and save them
           print("reset")
           cur_frame = st_frame+i+1
           print(cur_frame)
           set_no = save_tracks(tracked_points,info,seq_no,cur_frame,set_no,window_length,save_path)
           #del(tracked_points[0])
           tracked_points = []

           st_frame+=1
           first_image = pre + str(st_frame).zfill(6) + '.png' 
           first_img = cv2.imread(first_image)

           key_points = detect_keypoints(first_img)
           tracked_points.append(key_points)


        #st_frame += 1

    # post pricessing to make it like hopkins
if __name__ == "__main__":
   
   run(start_frame = int(sys.argv[3]),window_length=int(sys.argv[2]),seq_no=int(sys.argv[1])) 
   info = parse_bbox()  
   read_bbox(info)
   print("done reading")
   input()
   #itetrate in the diretory
   # call run on each diretory


print("start")
run()
input()
filename = "./image_02/0000/000048.png"
filename2 = "./image_02/0000/000049.png"

img = cv2.imread(filename)
img2 = cv2.imread(filename2)

kp = detect_keypoints(img) 
print("sift ")
print(len(kp))
kp1,st = track_points(img,img2,kp)
im2 =img 
cv2.drawKeypoints(img,kp,im2)

cv2.imwrite("sift.png",im2)


img = cv2.imread(filename)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

print("corners ")
print(len(corners))

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


cv2.imwrite("corners.png",img)

