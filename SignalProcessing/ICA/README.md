# ICA
## ICA_Experiment
### 目的
* ICAが苦手とするもの・得意とするものの差別化を行う
* sckit-learnのICAの特徴を掴む  

### 試験内容
* チュートリアルどおり
    * https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py  
    ここのコード(信号の部分)を書き換えて試験していく
* 周波数の違う正弦波
* 周波数の同じ正弦波と余弦波
* 同一信号を重ね合わせているときのICA
    * 同一信号2つと異なる信号のとき
    * 同一信号で微小な位相差あり

* バイアス成分が乗った信号  
    あるベースライン信号に加算する形で残り2つが存在する場合のこと
    * 長時間信号における変動全体でかけた場合
