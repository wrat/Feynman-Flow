#!/usr/bin/env python
from pathlib import Path

def getimgfiles(stem):
    stem = Path(stem)
    path = stem.parent
    name = stem.name
    exts = ['.ppm','.bmp','.png','.jpg']
    for ext in exts:
        pat = "{0}.*{1}".format(name,ext)
        print("searching {0}/{1}".format(path,pat))
        flist = sorted(path.glob(pat))
        if flist:
            break

    if not flist:
        raise FileNotFoundError("no files found under {0} with {1}".format(stem,exts))

    print("analyzing {0} files {1}.*{2}".format(len(flist),stem,ext))

    return flist,ext
