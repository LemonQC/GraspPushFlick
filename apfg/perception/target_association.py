"""Sensor-only temporal colored-3D target association, with explicit uncertainty.

Only initial task ROI, calibrated RGBD and proprioception may enter this module.
No asset ID, simulator pose, instance mask or evaluation callback is accepted.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter
def rgb_to_hsv(rgb):
    rgb=np.asarray(rgb,float);v=rgb.max(-1);lo=rgb.min(-1);delta=v-lo;s=delta/np.maximum(v,1e-12);h=np.zeros_like(v)
    for axis,offset in [(0,0),(1,2),(2,4)]:
        take=(rgb[...,axis]==v)&(delta>1e-12)
        h[take]=(offset+(rgb[..., (axis+1)%3]-rgb[..., (axis+2)%3])[take]/delta[take])/6
    return np.stack([h%1,s,v],axis=-1)

VERSION='rgbd_temporal_association_v2'

def camera_cloud(c):
    z=np.asarray(c['depth_m']);y,x=np.indices(z.shape)
    p=np.stack([(x-c['K'][0,2])*z/c['K'][0,0],(y-c['K'][1,2])*z/c['K'][1,1],z],axis=-1)
    return p@c['T_world_optical'][:3,:3].T+c['T_world_optical'][:3,3]

def colored_components(points,colors,radius=.010):
    if not len(points):return []
    pairs=cKDTree(points).query_pairs(radius,output_type='ndarray')
    # Color boundary limits growth across touching objects; disconnected texture
    # patches remain separate hypotheses rather than oracle instance merging.
    if len(pairs):pairs=pairs[np.linalg.norm(colors[pairs[:,0]]-colors[pairs[:,1]],axis=1)<.32]
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False)
    return [np.flatnonzero(labels==k) for k in np.unique(labels) if np.sum(labels==k)>=8]

def observed_overhang_band(points):
    """Separate an observed broad body from a narrower lower support.

    Pure geometric hypothesis, not an instance oracle. Both end strips must
    show a common elevated underside; isotropic/insufficient shapes stay whole.
    """
    if len(points)<60:return points,None
    lo,hi=np.quantile(points,.02,axis=0),np.quantile(points,.98,axis=0);extent=hi-lo;axis=int(np.argmax(extent[:2]))
    if extent[axis]<.10 or extent[axis]/max(extent[1-axis],.004)<1.5:return points,None
    ends=[points[points[:,axis]<lo[axis]+.028],points[points[:,axis]>hi[axis]-.028]]
    if any(len(e)<20 for e in ends):return points,None
    edge_lo=min(float(np.quantile(e[:,2],.03)) for e in ends);edge_hi=max(float(np.quantile(e[:,2],.97)) for e in ends)
    if edge_lo-lo[2]>.02 and edge_hi-edge_lo<.78*extent[2]:
        threshold=edge_lo-.006
        return points[points[:,2]>=threshold],float(threshold)
    return points,None


class TargetAssociation:
    def __init__(self,hint):
        if hint.get('type')!='initial_rgbd_region':raise ValueError('Explicit initial RGBD region required')
        self.roi=np.asarray(hint['world_xy_region'],float);self.template=None;self.center=None;self.last_points=np.empty((0,3));self.records=[];self.last_key=None;self.cached=None;self.shape_reference=None;self.motion_class="STATIC"
    def set_action_motion(self,kind):
        if kind not in ('ACTIVE','PUSH','FLIP','GRASP','STATIC'):raise ValueError('Unknown observed action class')
        self.motion_class=kind;self.last_key=None
    def apply_motion_prior(self,delta):
        # Commanded displacement is available to every policy, unlike GT motion.
        # This predicts a search center, never a mask or object identity.
        delta=np.asarray(delta,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('Finite commanded XYZ displacement required')
        if self.center is not None:self.center=self.center+delta
        if len(self.last_points):self.last_points=self.last_points+delta
        self.last_key=None
    def _appearance(self,colors):
        hsv=rgb_to_hsv(np.clip(colors,0,1));h=np.minimum((hsv[:,0]*18).astype(int),17);s=np.minimum((hsv[:,1]*5).astype(int),4)
        return self.template[h,s]
    def update(self,rgbd,time_s,exclude=None):
        # Repeated observe() at one physical instant must not adapt repeatedly.
        key=(float(time_s),tuple(np.asarray(rgbd[n]['T_world_optical']).tobytes() for n in sorted(rgbd)))
        if key==self.last_key:return self.cached
        self._pending_key=key
        self._extrinsics={n:np.asarray(c['T_world_optical']).tolist() for n,c in rgbd.items()}
        arrays={};all_p=[];all_c=[]
        for name,c in rgbd.items():
            p=camera_cloud(c);z=np.asarray(c['depth_m']);valid=np.isfinite(p).all(-1)&np.isfinite(z)&(z>.005)&(z<2)&(p[...,0]>.09)&(p[...,0]<.84)&(abs(p[...,1])<.44)&(p[...,2]>.008)&(p[...,2]<.55)
            ids=np.flatnonzero(valid);xyz=p.reshape(-1,3)[ids];rgb=np.asarray(c['rgb'])[...,:3].reshape(-1,3)[ids]/255.
            if exclude is not None:
                keep=exclude(xyz);ids=ids[keep];xyz=xyz[keep];rgb=rgb[keep]
            arrays[name]=(ids,xyz,rgb,z.shape);all_p.append(xyz);all_c.append(rgb)
        p=np.concatenate(all_p);c=np.concatenate(all_c)
        if len(p):
            _,idx=np.unique(np.floor(p/.004).astype(int),axis=0,return_index=True);p=p[idx];c=c[idx]
        pairs=cKDTree(p).query_pairs(.007,output_type='ndarray') if len(p) else np.empty((0,2),int)
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(p),len(p)))
        _,surface_labels=connected_components(graph,directed=False)
        initial=self.template is None
        if initial:
            a,b,d,e=self.roi;seed=(p[:,0]>=a)&(p[:,0]<=b)&(p[:,1]>=d)&(p[:,1]<=e)
            if seed.sum()<12:return self._result(arrays,np.empty((0,3)),time_s,'LOST',[],initial)
            # Bootstrap complete visible geometry through separated 3D surfaces,
            # rather than learning only the neutral top face inside a tiny ROI.
            # Contacting same-color objects can still be ambiguous, never GT-split.
            pairs=cKDTree(p).query_pairs(.007,output_type='ndarray')
            graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(p),len(p)))
            _,labels=connected_components(graph,directed=False)
            votes=np.bincount(labels[seed]);component=int(votes.argmax());grown=labels==component
            if np.ptp(p[grown],axis=0).max()<.26 and grown.sum()>=seed.sum()/2:
                _,band=observed_overhang_band(p[grown])
                seed=grown if band is None else grown&(p[:,2]>=band)
            hsv=rgb_to_hsv(c[seed]);hist,_,_=np.histogram2d(hsv[:,0],hsv[:,1],bins=(18,5),range=((0,1),(0,1)))
            hist=gaussian_filter(hist,(.7,.4),mode=('wrap','nearest'));self.template=hist/max(hist.max(),1.)
            self.center=np.median(p[seed],axis=0)
            self.shape_reference=np.maximum(np.linalg.eigvalsh(np.cov(p[seed].T)),1e-5)
        likelihood=self._appearance(c);valid=(likelihood>.16)&(np.linalg.norm(p-self.center,axis=1)<.34)
        cp=p[valid];cc=c[valid];cl=likelihood[valid];hypotheses=[];vl=surface_labels[valid]
        # Rejoin colored patches only when the measured 3D surface connects.
        # Two equally colored but separated objects remain different hypotheses.
        groups=[]
        for label in np.unique(vl):
            ix=np.flatnonzero(vl==label)
            if len(ix)<8:continue
            if np.ptp(p[surface_labels==label],axis=0).max()<.26:
                full,band=observed_overhang_band(p[surface_labels==label])
                if band is not None:ix=ix[cp[ix,2]>=band]
                if len(ix)>=8:groups.append((ix,label))
            else:
                for part in colored_components(cp[ix],cc[ix]):groups.append((ix[part],None))
        for ix,label in groups:
            points=cp[ix];center=np.median(points,axis=0);distance=float(np.linalg.norm(center-self.center));appearance=float(np.mean(cl[ix]))
            shape=np.maximum(np.linalg.eigvalsh(np.cov(points.T)),1e-5)
            shape_similarity=float(np.exp(-.65*np.mean(abs(np.log(shape/self.shape_reference)))))
            score=appearance*np.exp(-distance/.16)*shape_similarity*min(1.,len(ix)/40)
            previous_overlap=None
            if not initial and len(self.last_points):
                full=points if label is None else observed_overhang_band(p[surface_labels==label])[0]
                previous_overlap=float(np.mean(cKDTree(full).query(self.last_points)[0]<.012))
                score*=(.6+.8*previous_overlap) if self.motion_class in ('PUSH','FLIP','GRASP') else (.25+1.5*previous_overlap)
            if initial:
                a,b,d,e=self.roi
                score*=1+2*np.mean((points[:,0]>=a)&(points[:,0]<=b)&(points[:,1]>=d)&(points[:,1]<=e))
            hypotheses.append({'center_m':center.tolist(),'points':len(ix),'appearance':appearance,'distance_m':distance,'shape_similarity':shape_similarity,'previous_visible_surface_overlap':previous_overlap,'score':float(score),'indices':ix,'surface_label':None if label is None else int(label)})
        hypotheses.sort(key=lambda a:a['score'],reverse=True);status='LOST';target=np.empty((0,3))
        if hypotheses:
            best=hypotheses[0];status='TRACKED'
            # Similar-colored separated neighbors cannot be silently assigned.
            if len(hypotheses)>1 and hypotheses[1]['score']>.86*best['score'] and np.linalg.norm(np.asarray(best['center_m'])-hypotheses[1]['center_m'])>.035:status='UNCERTAIN'
            if best['score']<.09:status='LOST'
            if status=='TRACKED':
                target=cp[best['indices']] if best['surface_label'] is None else observed_overhang_band(p[surface_labels==best['surface_label']])[0];self.center=np.median(target,axis=0);self.last_points=target
        return self._result(arrays,target,time_s,status,hypotheses,initial)
    def _result(self,arrays,target,time_s,status,hypotheses,initial):
        masks={};tree=cKDTree(target) if len(target) else None
        for name,(ids,points,colors,shape) in arrays.items():
            mask=np.zeros(np.prod(shape),bool)
            if tree is not None:mask[ids[tree.query(points)[0]<.008]]=True
            masks[name]=mask.reshape(shape)
        record={'version':VERSION,'implementation_revision':6,'action_motion_class':self.motion_class,'time_s':float(time_s),'status':status,'initial_cue_consumed':bool(initial),'center_m':None if not len(target) else np.median(target,axis=0).tolist(),'target_voxels':len(target),'hypotheses':[{k:v for k,v in h.items() if k!='indices'} for h in hypotheses[:5]],'camera_extrinsics':self._extrinsics}
        self.records.append(record);self.last_key=self._pending_key
        self.cached=(masks,target,record);return self.cached
