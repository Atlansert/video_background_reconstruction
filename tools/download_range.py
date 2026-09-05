import argparse, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('url'); ap.add_argument('out'); ap.add_argument('--size',type=int,required=True); ap.add_argument('--chunk',type=int,default=64*1024*1024); ap.add_argument('--workers',type=int,default=8); a=ap.parse_args()
    n=(a.size+a.chunk-1)//a.chunk; Path=a.out
    os.makedirs(os.path.dirname(Path) or '.',exist_ok=True)
    with open(Path,'wb') as f: f.truncate(a.size)
    def get(i):
        lo=i*a.chunk; hi=min(a.size-1,(i+1)*a.chunk-1)
        for retry in range(8):
            try:
                r=requests.get(a.url,headers={'Range':f'bytes={lo}-{hi}'},timeout=(20,180)); r.raise_for_status(); data=r.content
                if len(data)!=(hi-lo+1): raise RuntimeError(f'chunk {i}: {len(data)} != {hi-lo+1}')
                with open(Path,'r+b') as f: f.seek(lo); f.write(data)
                return i,len(data)
            except Exception:
                if retry==7: raise
                time.sleep(2*(retry+1))
    done=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(get,i) for i in range(n)]
        for fut in as_completed(fs):
            i,b=fut.result(); done+=1; print(f'{done}/{n} chunk {i} ({b/1e6:.1f} MB)',flush=True)
if __name__=='__main__': main()
