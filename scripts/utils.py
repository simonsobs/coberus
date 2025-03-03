import os
from pixell import enmap, bunch
from orphics import io
import numpy as np

act_tags = [f'{t}_{x}' for x in  ['pa4_f220',
                             'pa4_f150',
                             'pa5_f090',
                             'pa5_f150',
                             'pa6_f090',
                             'pa6_f150'] for t in ['night','daydeep','daywide']]

paths = io.config_from_yaml('paths_local.yaml')

planck_root = paths['planck_root']
act_root = paths['act_root']
out_root = paths['out_root']
mask_root = paths['mask_root']
sim_root = paths['sim_root']
cmb_sim_fname = lambda simid,iset=0: f"{sim_root}/fullskyLensedUnabberatedCMB_alm_set0{iset}_{simid:05d}.fits"

planck_tags = ['030','044','070','100','143','217','353','545']

# # Quick plots
# def plot(imap,tag,ind,mtype='map',**kwargs):
#     io.hplot(imap,f'{out_root}/wavelet_{mtype}_{tag}_scale_{ind}',mask=0,**kwargs)

def parse_tags(t):
    if t is None:
        tags = act_tags + planck_tags
        tags.remove('daywide_pa4_f220')
        tags.remove('daywide_pa6_f090')
        tags.remove('daywide_pa6_f150')
        tags.remove('030')
        tags.remove('044')
        tags.remove('070')
        tags.remove('545')
    else:
        tags = t.split(',')
    return tags

def get_lpeaks(basis):
    if basis=='lensmode':
        lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000., 5000., 6000.]
    elif basis=='szmode':
        lpeaks = np.append(np.append([0.,100.,500.,800.,1000.,2000.,3000.,4000.],np.arange(6000.,24000.,4000.)),[30000.])
    elif basis=='debug':
        lpeaks = [800.,1000.,2000.,3000.,4000.]
    return lpeaks

def get_properties(yaml_file,tags):
    lmins = []
    lmaxs = []
    fwhms = []

    c = io.config_from_yaml(yaml_file)
    print(c)
    def _clean(item):
        if item=='None': return None
        return float(item)

    d = {}
    for tag in tags:
        d[tag] = {}
        if ('night' in tag) or ('daywide' in tag) or ('daydeep' in tag):
            freq = tag.split('_')[-1]
            key = f'act_{freq}'
            d[tag]['exp'] = 'act'
        else:
            key = tag
            d[tag]['exp'] = 'planck'
            d[tag]['sim_noise'] = c[key]['sim_noise']
        lmin = _clean(c[key]['lmin'])
        lmax = _clean(c[key]['lmax'])
        fwhm = _clean(c[key]['beam'])
        lmins.append(lmin)
        lmaxs.append(lmax)
        fwhms.append(fwhm)
        
    return lmins, lmaxs, fwhms, d

def downgrade(omap,dfact,**kwargs):
    if (omap is None) or dfact==1: return omap
    return enmap.downgrade(omap,dfact,**kwargs)

def get_filename(tag,maptype='map',splitnum=None,srcfree=True):
    if not(maptype in ['map','ivar','mask60','mask70','mask80']): raise ValueError
    if maptype[:4]=='mask':
        # We get the name of the patch from the tag, and use night if it's Planck
        patch = tag.split('_')[0] if tag in act_tags else 'night'
        fname = os.path.join(f'{mask_root}',f"dr6v4_lensing_20240919_{patch}_enhanced_mask_{maptype[-2:]}.fits")
        return fname
        
    if tag in act_tags:
        fname = act_root
        maxsplits = 4
    elif tag in planck_tags:
        fname = planck_root
        maxsplits = 2
    else:
        raise ValueError(f"Not a recognized array tag. Choose from {act_tags} or {planck_tags}.")
    if not(splitnum is None):
        if splitnum<=0: raise ValueError
        splitid = splitnum if (tag in act_tags) else (splitnum+1)
        if (splitnum+1)>maxsplits: raise ValueError
        coaddstr = f"set{splitid}" if (tag in act_tags) else f"split{splitid}"
    else:
        coaddstr = "coadd"

    if tag in act_tags:
        if srcfree and maptype=='map':
            maptype = 'map_srcfree'
        fname = os.path.join(fname,f"cmb_{tag}_3pass_4way_{coaddstr}_{maptype}.fits")
    else:
        if srcfree and maptype=='map':
            maptype = 'srcfree'
        fname = os.path.join(fname,f"planck_npipe_{tag}_{coaddstr}_{maptype}.fits")
        
    if not(os.path.isfile(fname)): raise FileNotFoundError
    return fname
    
    
    
