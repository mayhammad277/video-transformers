import torch
def get_prediction (sent,tokenizer,model): 
    
     token_ids = tokenizer.encode(sent,max_length=512, return_tensors='pt') 
     masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero() 
     #masked_position = (token_ids.squeeze() == tokenizer.mask_token_id)
     masked_pos = [mask.item() for mask in masked_position ] 
  
     with torch.no_grad(): 
         output = model(token_ids) 
  
     last_hidden_state = output[0].squeeze() 
  
     list_of_list =[] 
     for index,mask_index in enumerate(masked_pos): 
         mask_hidden_state = last_hidden_state[mask_index] 
         idx = torch.topk(mask_hidden_state, k=5, dim=0)[1] 
         words = [tokenizer.decode(i.item()).strip() for i in idx] 
         list_of_list.append(words) 
         #print ("Mask ",index+1,"Guesses : ",words)
     print('all mask predictions',list_of_list) 
     return list_of_list
     #best_guess = "" 
     #for j in list_of_list: 
     #    best_guess = best_guess+" "+j[0] 
     #    #print('best_guess',best_guess) 
     #return best_guess 

