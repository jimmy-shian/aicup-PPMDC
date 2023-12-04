import os
import shutil
import configparser
import pickle
from datetime import datetime
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import transformers
from transformers import GPT2Tokenizer, GPT2Model,  GPT2LMHeadModel #, GPT2AdapterModel
# from adapter_transformers import adapter_hub
import adapters
from adapters import GPT2AdapterModel

from transformers import GPTJForCausalLM
from utils.custom_dataset import CustomDataset
from utils.pytorchtools import EarlyStopping

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 回傳 data 中最大的 index
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

def train_epoch(model, device, train_dataloader, optimizer, scheduler,
                epoch):
    # model.train()
    model.train_adapter("token_classification") #改這行之後他開始可以動作

    device = device
    epoch_start_time = datetime.now()
    total_loss = 0  # 紀錄整個 epoch 的 loss 總和

    # epoch_correct_num: 每個 epoch 中, 預測正確的 word 數量
    # epoch_total_num: 每個 epoch 中, 預測的所有 word 數量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # catch cuda out of memory exception (依據 memory 大小調整 batch_size)
        batch_acc = 0
        bestacc = 0
        cc = 0
        try:
            while(batch_acc < 1.00): #這裡多用一個無限迴圈，他多跑幾次?不確定有沒有用，因為沒有成功跑完過
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                # 統計該 batch 中預測 token 的正確數與總數
                batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=int(config['parameters']['ignore_index']))
                # 統計該 epoch 中預測 token 的正確數與總數
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                # 計算該 batch 的 accuracy
                batch_acc = batch_correct_num / batch_total_num

                total_loss += loss.item()
                if int(config['parameters']['gradient_accumulation_steps']) > 1:
                    loss = loss / int(config['parameters']['gradient_accumulation_steps'])

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['parameters']['max_grad_norm'])

                # 進行一定 step 的梯度累計後更新參數
                if (batch_idx + 1) % int(config['parameters']['gradient_accumulation_steps']) == 0:
                    # 更新參數
                    optimizer.step()
                    # 更新 learning rate
                    scheduler.step()
                    # 清空梯度資訊
                    optimizer.zero_grad()
                print(f'\t batch {batch_idx + 1} of epoch {epoch + 1},loss {loss},batch_acc {batch_acc},lr {scheduler.get_lr()}')
                cc += 1  
                if batch_acc > bestacc:
                    bestacc = batch_acc
                with open("show_detail.txt", 'a+') as file:
                    file.write((f'\t{cc:03}: Batch {batch_idx + 1} of epoch {epoch + 1}, loss {loss}, batch_acc {batch_acc}, lr {scheduler.get_lr()}\n') )
                if ((batch_idx + 1) % int(config['parameters']['log_step']) == 0 and batch_acc > 0.98) or (cc > 10 and batch_acc >=bestacc)  or (cc > 30):
                    loss = loss.item() * int(config['parameters']['gradient_accumulation_steps'])
                    print(f'Batch {batch_idx + 1} of epoch {epoch + 1}, loss {loss}, batch_acc {batch_acc}, lr {scheduler.get_lr()}')
                    with open("show_detail.txt", 'a+') as file:
                        file.write((f'{cc:03}: Batch {batch_idx + 1} of epoch {epoch + 1}, loss {loss}, batch_acc {batch_acc}, lr {scheduler.get_lr()}\n') )
                    break
            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                print(str(exception))
                raise exception

    # 紀錄當前 epoch 的平均 loss 與 accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print(f'Epoch {epoch + 1}: loss {epoch_mean_loss}, predict_acc {epoch_mean_acc}')

    # save model
    print('Saving model for epoch {}'.format(epoch + 1))
    model_path = os.path.join(config['finetune_path']['save_model_path'], 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)

    # model.save_adapter("path/to/adapter", "poem")
    model_path2 = os.path.join(config['finetune_path']['save_model_path'], 'adapter_epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path2):
        os.mkdir(model_path2)
    model.save_adapter(model_path, "token_classification")
# ====================================================================
    pre_model_path = os.path.join(config['finetune_path']['save_model_path'], 'epoch{}'.format(epoch))
    if os.path.exists(pre_model_path):
        shutil.rmtree(pre_model_path)
    pre_model_path2 = os.path.join(config['finetune_path']['save_model_path'], 'adapter_epoch{}'.format(epoch + 1))
    if os.path.exists(pre_model_path2):
        shutil.rmtree(pre_model_path2)

    print(f'Epoch {epoch + 1} finished')
    epoch_finish_time = datetime.now()
    print(f'Time for one epoch: {epoch_finish_time - epoch_start_time}')
    return epoch_mean_loss, epoch_mean_acc


def val_epoch(model, device, val_dataloader, epoch):
    print("Start Validating")
    model.eval()
    device = device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    epoch_start_time = datetime.now()
    total_loss = 0
    # catch cuda out of memory exception (依據 memory 大小調整 batch_size)
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(val_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 紀錄當前 epoch 的平均 loss
            epoch_mean_loss = total_loss / len(val_dataloader)
            print(f'validate epoch {epoch+1}: loss {epoch_mean_loss}')
            epoch_finish_time = datetime.now()
            print(f'time for validating one epoch: {epoch_finish_time - epoch_start_time}')
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            print(str(exception))
            raise exception

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config/configuration.ini', encoding='utf8')

    '''
    Encode the training set and output the pickle file for model training
    '''
    # 初始化 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
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

    # 讀取訓練集與測試集
    train_path = os.path.join(os.getcwd(), config['finetune_path']['train_path'])
    train_path = os.path.join(train_path, 'train.pkl')
    test_path = os.path.join(os.getcwd(), config['finetune_path']['test_path'])
    test_path = os.path.join(test_path, 'test.pkl')

    '''
    GPT-2 model training
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if (torch.cuda.is_available()):
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('device=' ,device)
    # Loading pre-trained model
    # model = GPT2Model.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

    # model = GPT2LMHeadModel.from_pretrained(config['inference_path']['finetuned_model_path'])
    model = GPT2AdapterModel.from_pretrained(config['inference_path']['finetuned_model_path'])

    # 添加適配器
    # 添加適配器
    model.add_adapter("token_classification")
    # model.load_adapter(config['inference_path']['finetuned_model_path2']) #第二次載入應該是要用這個

    # 將適配器添加到指定的位置
    model.set_active_adapters("token_classification")

    # 訓練適配器
    # model.train_adapter("token_classification")
    # model.set_active_adapters("token_classification")

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    assert model.config.vocab_size == len(tokenizer)
    if ((device != 'cpu') and (torch.cuda.device_count() > 1)):
        model = DataParallel(model).cuda()
        print(f'Use GPU {device} to train')
    
    # Loading dataset
    with open(train_path, "rb") as f:
        input_list = pickle.load(f)
    # val_num = int(len(input_list)*0.2)
    input_list_train = input_list
    with open(test_path, "rb") as f:
        input_list_test = pickle.load(f)
    input_list_val = input_list_test
    train_dataset = CustomDataset(input_list_train, int(config['parameters']['max_len']))
    val_dataset = CustomDataset(input_list_val, int(config['parameters']['max_len']))
    train_dataloader = DataLoader(
            train_dataset, batch_size=int(config['parameters']['batch_size']), shuffle=True, num_workers=int(config['parameters']['num_workers']), collate_fn=collate_fn,
            drop_last=True
        )
    val_dataloader = DataLoader(
            val_dataset, batch_size=int(config['parameters']['batch_size']), shuffle=True, num_workers=int(config['parameters']['num_workers']), collate_fn=collate_fn, 
            drop_last=True
        )
    if not os.path.exists(config['finetune_path']['save_model_path']):
        os.makedirs(config['finetune_path']['save_model_path'])
    early_stopping = EarlyStopping(int(config['parameters']['patience']), verbose=True)
    t_total = len(train_dataloader) # gradient_accumulation_steps * epochs
    optimizer = transformers.AdamW(model.parameters(), lr=float(config['parameters']['lr']), eps=float(config['parameters']['eps']))
    scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config['parameters']['warmup_steps']), num_training_steps=t_total
        )
    print('Starting Training')
    # 紀錄城市的運行開始時間
    start_time = datetime.now()
    
    train_losses, val_losses = list(), list()
    best_val_loss = 10**4 # initial validation loss
    best_epoch = 1 # initial best epoch
    for epoch in range(int(config['parameters']['epochs'])):
        # train
        train_loss, train_acc = train_epoch(
                model=model, device=device, train_dataloader=train_dataloader,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch)
        train_losses.append(train_loss)
        # validate
        val_loss = val_epoch(
                model=model, device=device, val_dataloader=val_dataloader,
                epoch=epoch)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pre_model_path = os.path.join(config['finetune_path']['save_model_path'], 'min_ppl_model')
            pre_model_path2 = os.path.join(config['finetune_path']['save_model_path'], 'best_adapter')
            if os.path.exists(pre_model_path):
                shutil.rmtree(pre_model_path)
            if os.path.exists(pre_model_path2):
                shutil.rmtree(pre_model_path2)
            best_epoch = epoch + 1
            print('Saving current best model for epoch {}'.format(best_epoch))
            model_path = os.path.join(config['finetune_path']['save_model_path'], 'min_ppl_model')
            model_path2 = os.path.join(config['finetune_path']['save_model_path'], 'best_adapter')
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            if not os.path.exists(model_path2):
                os.mkdir(model_path2)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

            model.save_adapter(model_path, "token_classification")

            model_info_path = os.path.join(model_path, 'model_info.txt')
            with open (model_info_path, 'w', encoding='utf8') as w:
                w.write(f'epoch: {best_epoch}\n')
                w.write(f'train_losses: {train_loss}\n')
                w.write(f'train_acc: {train_acc}\n')
                w.write(f'validate_losses:{val_loss}')
        if int(config['parameters']['patience']) == 0:
            continue
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Training Finished')
        print(f'train_losses:{train_losses}')
        print(f'validate_losses:{val_losses}')
        # 紀錄城市的運行結束時間
    end_time = datetime.now()

    # 計算運行時間
    run_time = end_time - start_time
    print('finish: ', run_time) 
    # finish:  8:42:45.435414
    # finish:  6:54:53.796904 Epch:50
