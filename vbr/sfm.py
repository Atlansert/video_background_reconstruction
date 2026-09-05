"""Classical monocular SfM fallback: SIFT tracking, essential poses, sparse triangulation."""
from pathlib import Path
import cv2, numpy as np

def reconstruct(frame_dir, output_dir, focal=None, max_frames=80):
    paths=sorted(Path(frame_dir).glob('*.jpg')); paths=[paths[i] for i in np.linspace(0,len(paths)-1,min(max_frames,len(paths)),dtype=int)]
    if len(paths)<2: raise RuntimeError('Need at least two frames')
    first=cv2.imread(str(paths[0]),cv2.IMREAD_GRAYSCALE); h,w=first.shape; f=float(focal or max(w,h)); K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]],float)
    feat=cv2.SIFT_create(nfeatures=4000); matcher=cv2.BFMatcher(cv2.NORM_L2)
    poses=[np.hstack((np.eye(3),np.zeros((3,1))))]; points=[]; colors=[]; prev=first; kp_prev,dprev=feat.detectAndCompute(prev,None)
    for n,p in enumerate(paths[1:],1):
        cur=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE); kp,dcur=feat.detectAndCompute(cur,None)
        if dprev is None or dcur is None: poses.append(poses[-1]); prev,kp_prev,dprev=cur,kp,dcur; continue
        pairs=matcher.knnMatch(dprev,dcur,k=2); good=[a for a,b in pairs if a.distance<.72*b.distance]
        if len(good)<12: poses.append(poses[-1]); prev,kp_prev,dprev=cur,kp,dcur; continue
        a=np.float32([kp_prev[m.queryIdx].pt for m in good]); b=np.float32([kp[m.trainIdx].pt for m in good]); E,inl=cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=.999,threshold=1.)
        if E is None: poses.append(poses[-1]); prev,kp_prev,dprev=cur,kp,dcur; continue
        _,R,t,mask=cv2.recoverPose(E,a,b,K); Pprev=poses[-1]; P=np.hstack((R,t))@np.vstack((Pprev,[0,0,0,1])); poses.append(P); valid=mask.ravel()>0; a=a[valid]; b=b[valid]
        if len(a)>5:
            X=cv2.triangulatePoints(K@Pprev,K@P,a.T,b.T); X=(X[:3]/X[3]).T; ok=np.isfinite(X).all(1)&(np.linalg.norm(X,axis=1)<100); points.append(X[ok]); rgb=cv2.imread(str(paths[n-1])); uv=a[ok].astype(int); uv[:,0]=np.clip(uv[:,0],0,w-1); uv[:,1]=np.clip(uv[:,1],0,h-1); colors.append(rgb[uv[:,1],uv[:,0],::-1])
        prev,kp_prev,dprev=cur,kp,dcur
    pts=np.concatenate(points) if points else np.empty((0,3)); col=np.concatenate(colors) if colors else np.empty((0,3),np.uint8); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); np.savez(out/'sfm.npz',poses=np.array(poses),intrinsics=K,points=pts,colors=col,frame_paths=np.array([str(p) for p in paths]))
    import open3d as o3d
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); cloud.colors=o3d.utility.Vector3dVector(col.astype(float)/255 if len(col) else np.empty((0,3))); o3d.io.write_point_cloud(str(out/'points.ply'),cloud)
    return out/'points.ply'
