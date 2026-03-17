import shutil, os
src = 'tokenizer_cache'
dst = 'model'
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    shutil.copy2(os.path.join(src,f), os.path.join(dst,f))
    print('Copied:', f)