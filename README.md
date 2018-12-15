

# Introduction

* main.cc : opencv 讀取 frame 並作偵測
* image : 測試用的，直接讀取 image

# 目前

在 x86 + 1070 上已經測試過，可以正常執行，但是線程的部分可能還要稍作調整，在 tx2 上好像還是不太穩定，

至於 tx2 之前讀取不到 NDArray 的問題不知道怎麽的成就突然可以了

# Todo



# 注意

# NDArray 用法

最好不要

NDArray nd;

nd = NDArray(Shape(3, 224, 224), ctx);

可以的話盡量

當下就創建好

NDArray nd(Shape(3, 224, 224), ctx);

或是

map <string, NDArray> nds;

nds["nd"] = NDArray(Shape(3, 224, 224), ctx);