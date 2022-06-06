import pandas as pd
import numpy as np

def inter(array):
    music_id = music_dict[int(array['music_id'])]
    user_id = int(array['user_id'])
    writer.write(str(user_id)+' '+str(music_id)+'\n')

def func_user(a):
    if a!=np.nan and int(a) in user_dict_half:
        return int(user_dict_half[int(a)])
    else: return -1

def func_music(a):
    if a!=np.nan and int(a) in music_dict:
        return int(music_dict[int(a)])
    else: return -1
def func_split(x):
    if x is not np.nan and x!='None' and x is not None:
        r=[]
        for a in x.strip().split('/'):
            r.append(entity_dict[a])
        return r
    else:
        return None
def func_playlist(x):
    if x is not np.nan and x!='None' and x is not None:
        r=[]
        for a in x.strip().split('-'):
            r.append(entity_dict[a])
        return r
    else:
        return None
def generate(array):
    mid=int(array['id'])
    singer=array['singer']
    album=array['album']
    composer=array['composer']
    author=array['author']
    ply=array['playlist']
    if singer is not None and singer!='None' and singer is not np.nan:
        for s in singer:
            writer_final.write(str(mid)+' '+str(0)+' '+str(int(s))+'\n')
    if album is not None and album!='None' and album is not np.nan :
        for s in album:
            writer_final.write(str(mid)+' '+str(1)+' '+str(int(s))+'\n')
    if composer is not None and composer!='None' and composer is not np.nan:
        for s in composer:
            writer_final.write(str(mid)+' '+str(2)+' '+str(int(s))+'\n')
    if author is not None and author!='None' and author is not np.nan:
        for s in author:
            writer_final.write(str(mid)+' '+str(3)+' '+str(int(s))+'\n')
    if ply is not None and ply!='None' and ply is not np.nan:
        for s in ply:
            writer_final.write(str(mid)+' '+str(4)+' '+str(int(s))+'\n')
if __name__=='__main__':
    mc = pd.read_csv('../data/music/original/music_comments.csv')
    user_dict={}
    cnt=0
    for u in mc['user_id']:
        if int(u) not in user_dict:
            user_dict[int(u)]=cnt
            cnt+=1

    user_dict_half = {}
    l=len(user_dict)
    cnt=0
    for u in user_dict:
        user_dict_half[u]=user_dict[u]
        cnt+=1
        if cnt>=l/2:
            break

    mc['user_id'] = mc['user_id'].map(func_user)
    mc_half = mc[mc['user_id']!=-1]

    music_dict={}
    cnt=0
    for m in mc_half['music_id']:
        if int(m) not in music_dict:
            music_dict[int(m)]=cnt
            cnt+=1

    writer=open('../data/music/original/interaction.txt','w',encoding='utf-8')

    mc_half.apply(inter,axis=1)

    md=pd.read_csv('../data/music/original/music_data.csv')
    md=md.drop(['name','comment_num','comment_users'],axis=1)
    md['id'] = md['id'].map(func_music)
    md_half = md[md['id']!=-1]

    entity_dict={}
    cnt=19870

    for a in md_half['singer']:
        if a and a is not np.nan and a is not None and a!='None':
            l=a.strip().split('/')
            for i in l:
                if i not in entity_dict:
                    entity_dict[i]=cnt
                    cnt+=1
    for a in md_half['album']:
        if a and a is not np.nan and a is not None and a!='None':
            l=a.strip().split('/')
            for i in l:
                if i not in entity_dict:
                    entity_dict[i]=cnt
                    cnt+=1
    for a in md_half['composer']:
        if a and a is not np.nan and a is not None and a!='None':
            l=a.strip().split('/')
            for i in l:
                if i not in entity_dict:
                    entity_dict[i]=cnt
                    cnt+=1
    for a in md_half['author']:
        if a and a is not np.nan and a is not None and a!='None':
            l=a.strip().split('/')
            for i in l:
                if i not in entity_dict:
                    entity_dict[i]=cnt
                    cnt+=1
    for a in md_half['playlist']:
        if a and a is not np.nan and a is not None and a!='None':
            l=a.strip().split('-')
            for i in l:
                if i not in entity_dict:
                    entity_dict[i]=cnt
                    cnt+=1
    md_half['singer']=md_half['singer'].map(func_split)
    md_half['album']=md_half['album'].map(func_split)
    md_half['composer']=md_half['composer'].map(func_split)
    md_half['author']=md_half['author'].map(func_split)
    md_half['playlist']=md_half['playlist'].map(func_playlist)

    writer_final = open('../data/music/original/kg_final.txt','w',encoding='utf-8')
    relation_dict={'singer':0,'album':1,'composer':2,'author':3,'playlist':4}
    md_half.apply(generate,axis=1)
    print('done')