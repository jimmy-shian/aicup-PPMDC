# aicup-PPMDC（Privacy Protection and Medical Data Standardization Competition）

decode.ipynb 是製作訓練資料，以及處理predict.txt轉answer.txt
---
aicup/data 是First_Phase_Text_Dataset + Validation_Release + Second_Phase_Dataset
 
> data_old 則是移除掉 Validation_Release
 
> data_old_new 是將 filexxx.txt的檔案多複製一次，讓資料型態1:1平衡

---
train.pkl/test.pkl 的文字檔是 train.txt (訓練資料)
test2.pkl 的文字檔是 test.txt (測試資料)
 
aicup/answer.txt 為原始答案(合集)

---
### 此程式碼為比賽生成時使用的模型與資料
> gpt2_finetune.py 是原本的訓練檔案(有模型檔 min_ppl_model )

> gpt2_inference.py 生成predict.txt

---
### 此程式碼為創新想法製作，但是效果沒有預期的提高，因此未採用

> gpt2_finetune_2.py 是新加入RLHF想法的檔案(需更改套件版本)

>> gpt2_finetune_2_v2.py 增加adapter，結果不適很好

> gpt2_inference_v2.py 使用adapter生成predict.txt(套件需更改，所以輸出上可能不太正確)
