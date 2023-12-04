import sys
import os
import pickle
import configparser
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datetime import datetime
import adapters
from adapters import GPT2AdapterModel  

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()回傳最後一維最大的top_k個element, 回傳為二維(values,indices)
        # 其他維度由模型自行判斷
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 對於topk以外的其他对于topk之外的element, logit值設為負無窮

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 對logits進行遞減排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def get_generate (query):
    response = []  # 根據context而生成的response
    input_ids = [tokenizer.cls_token_id]
    # input_ids.extend(tokenizer.encode(query, add_special_tokens=False)[:512]) # 取 encode 上限數量
    input_ids += tokenizer.encode(query, add_special_tokens=False)
    # input_ids.append(tokenizer.encode(query, add_special_tokens=False)) # 取 encode 上限數量
    input_ids.append(tokenizer.sep_token_id)
    input_ids = torch.tensor(input_ids).long().to(device)
    input_ids = input_ids.unsqueeze(0)
    input_ids_update = input_ids
    for _ in range(int(config['parameters']['max_len'])):
    # for _ in range(len(input_ids_update)):
        outputs = model(input_ids=input_ids_update)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        # 對於已生成的結果, generated中的每個token增加一個重複懲罰, 以降低其生成機率
        for id in set(response):
            next_token_logits[id] /= float(config['parameters']['repetition_penalty'])
        next_token_logits = next_token_logits / float(config['parameters']['temperature'])
        # 將[UNK]的機率設為無窮小, 其模型的預測結果不會是[UNK]這個token
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=int(config['parameters']['topk']), top_p=int(config['parameters']['topp']))

        softmax = F.softmax(filtered_logits, dim=-1)

        # torch.argmax表示抽取權重最高的element
        next_tokens = torch.tensor([torch.argmax(softmax).item()]).to(device)

        token = torch.tensor([next_tokens.item()]).to(device)
        if token == (tokenizer.sep_token_id ):  # 遇到[SEP]則表示response生成結束  or 'Ċ' or '&amp;'
            input_ids_update = torch.cat((input_ids_update, token.unsqueeze(0)), dim=1)
            break
        response.append(token.item())
        input_ids_update = torch.cat((input_ids_update, token.unsqueeze(0)), dim=1)
    generate_text = tokenizer.convert_ids_to_tokens(response)
    generate_text = "".join(generate_text) #.strip()
    return generate_text

if __name__ == '__main__':
    # sys.setrecursionlimit(8735 * 2080 + 10)

    config = configparser.ConfigParser()
    config.read('config/configuration.ini', encoding='utf8')

    # 初始化 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    special_tks = list()
    special_tks.append('#@#') # code separator
    special_tks.append('<NL>') # code separator
    # special_tks.append('<None>') # code separator
    special_tks.append('[CLS]') # [CLS] 開頭
    special_tks.append('[PAD]') # [PAD] 中間
    special_tks.append('[SEP]') # [SEP] 結束
    tokenizer.add_special_tokens({"additional_special_tokens": special_tks})
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if (torch.cuda.is_available()):
        device = 'cuda:0'
    else:
        device = 'cpu'
    # 紀錄城市的運行開始時間
    start_time = datetime.now()

    # Loading pre-trained model
    # model = GPT2LMHeadModel.from_pretrained(config['inference_path']['finetuned_model_path'])
    model = GPT2AdapterModel.from_pretrained(config['inference_path']['finetuned_model_path'])

    # 添加適配器
    # model.load_adapter(config['inference_path']['finetuned_model_path2']) #第二次載入應該是要用這個

    model = model.to(device)
    model.eval()
    count = 0
    inference_path = os.path.join(config['inference_path']['save_inference_path'], 'predict.txt')
    except_path = os.path.join(config['inference_path']['except_path'], 'except.txt')
    with open(inference_path, 'w+', encoding='utf8') as w:
        test_path = os.path.join(config['inference_path']['test_path'], 'test2.pkl')
        file = open(test_path, 'rb')
        test_data = pickle.load(file)
        intint = 0
        for data in test_data:
            count += 1
            generate_text = tokenizer.decode(data)
            process_text = generate_text.split('[SEP]')[0].replace('[CLS]', '').strip()
            try:
                # print(f'\nQuery: {process_text}')
                response = get_generate(process_text)
                response = response.replace('#@#','\n').replace('ĉ', '\t').replace('Ċ', ' ').replace('<NL>', ' ')
                response = response.replace('Ġ', ' ').strip()
                print(f'\nGPT:\n{response}')
                w.write(response)
                # w.write('\t')
                # w.write(process_text)
                if count == 15:
                    w.write('\n=====================')
                    count = 0
                w.write('\n')
            except:
                with open (except_path, 'a+', encoding='utf8') as e:
                    e.write(process_text)
                    e.write('\n')
            intint += 1
            print(f"第{intint}筆")
    # 紀錄城市的運行結束時間
    end_time = datetime.now()

    # 計算運行時間
    run_time = end_time - start_time
    print('finish: ', run_time) 
    # finish:  2:01:58.944980 614/4
    # finish:  2:34:42.739631 540/8
    # finish:  1:18:48.249600 540/6
    # finish:  1:20:50.272822 540/10
    # finish:  1:20:27.375799 540/10 -NL None
