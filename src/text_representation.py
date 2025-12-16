import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

#tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base-fin")
#model = BertModel.from_pretrained("Langboat/mengzi-bert-base-fin")
tokenizer = BertTokenizer.from_pretrained("./Langboat/")
model = BertModel.from_pretrained("./Langboat")

class texts(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, texts_list):
        """
        """
        self.texts_list = texts_list

    def __len__(self):
        return len(self.texts_list)

    def __getitem__(self, idx):
        
        return self.texts_list[idx]
    
def texts_dataloader(dataset,batch_size,tokenizer):
    def collate_fn(batch_texts):
        texts = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt",max_length=512)
        input_ids = texts['input_ids']
        attention_mask = texts['attention_mask']
        token_type_ids = texts['token_type_ids']
        return input_ids,attention_mask,token_type_ids
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=collate_fn,drop_last=False)

df = pd.read_csv(r'./processed_data/text_data.csv',encoding='utf-8')
df= df.fillna(" ")
title_dataset = texts(df['text'].tolist())
title_dataloader = texts_dataloader(title_dataset,batch_size=512,tokenizer=tokenizer)

title_embs = []
with torch.no_grad():
    model.cuda()
    model.eval()
    for step,(input_ids,attention_mask,token_type_ids) in tqdm(enumerate(title_dataloader)):
        output = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda())
#         print(output.shape)
        titles_emb = output.last_hidden_state.mean(dim=1).detach()
#         print(titles_emb.shape)
        title_embs.append(titles_emb.cpu().numpy())
        if step%10==0:
            print(step)
title_embs_array = np.vstack(title_embs)
title_embs_df = pd.DataFrame(data=title_embs_array,columns=['emb'+str(i) for i in range(768)],index=df.index)
title_embs_df.to_csv('./result/Mengzi_base_emb.csv')