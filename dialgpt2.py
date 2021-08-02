import os
import time
import datetime

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

import transformers as tr

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="microsoft/DialoGPT-small", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer(txt + tokenizer.eos_token, truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]

train_file = 'data/male_corpus_CMV_train.csv'
test_file = 'data/male_corpus_CMV_test.csv'

df = pd.read_csv(train_file)  
df.dropna(inplace=True) #remove NA values
train_sentences = df.sentence.copy()

df = pd.read_csv(test_file)  
df.dropna(inplace=True) #remove NA values
test_sentences = df.sentence.copy()


tokenizer = tr.AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", pad_token='[PAD]')

train_dataset = GPT2Dataset(train_sentences, tokenizer, max_length=768)
test_dataset = GPT2Dataset(test_sentences, tokenizer, max_length=768)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(test_dataset)))


# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
batch_size = 2

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = SequentialSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# instantiate the model
model = tr.AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print(tokenizer.batch_decode([0, 50257]))
#model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
#device = torch.device("cuda")
device = torch.device("cpu")
#model.cuda()

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# some parameters I cooked up that work reasonably well

epochs = 1
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = tr.AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = tr.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


training_stats = []

model = model.to(device)
model.resize_token_embeddings(len(tokenizer))
assert model.transformer.wte.weight.shape[0] == len(tokenizer)
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_test_loss = total_eval_loss / len(test_dataloader)
     

    print("  Test Loss: {0:.2f}".format(avg_test_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Test Loss': avg_test_loss,
        }
    )

print("")
print("Training complete!")


output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# #tokenizer = tr.GPT2Tokenizer.from_pretrained("gpt2")

# print('vocabulary size: %d, max squence length: %d' % (tokenizer.vocab_size, tokenizer.model_max_length))
# print('tokenize sequence "My mother would like to make a donation":', tokenizer('My mother would like to make a donation'))

# data_collator = tr.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# #model = tr.GPT2LMHeadModel.from_pretrained('gpt2')
# model = tr.AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")





# train_dataset = tr.TextDataset(
#     tokenizer=tokenizer,
#     file_path=train_path,
#     block_size=128)
     
# test_dataset = tr.TextDataset(
#     tokenizer=tokenizer,
#     file_path=test_path,
#     block_size=128)

# training_args = tr.TrainingArguments(
#     output_dir = 'male', # the output directory for the model predictions and checkpoints
#     overwrite_output_dir = True, # overwrite the content of the output directory
#     per_device_train_batch_size = 32, # the batch size for training
#     per_device_eval_batch_size = 32, # the batch size for evaluation
#     learning_rate = 5e-5, # defaults to 5e-5
#     num_train_epochs = 3, # total number of training epochs to perform
# )

# trainer = tr.Trainer(
#     model = model,
#     args = training_args,
#     data_collator=data_collator,
#     train_dataset = train_dataset,
#     eval_dataset = test_dataset
# )

# trainer.train()

# trainer.save_model()


# def construct_conv(row, tokenizer, eos = True):
#     flatten = lambda l: [item for sublist in l for item in sublist]
#     conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
#     conv = flatten(conv)
#     return conv

# class ConversationDataset(Dataset):
#     def __init__(self, tokenizer: tr.AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium"), df, block_size=512):

#         block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

#         directory = 'model'
#         cached_features_file = os.path.join(
#             directory, "cached_lm_" + str(block_size)
#         )

#         if os.path.exists(cached_features_file):
#             logger.info("Loading features from cached file %s", cached_features_file)
#             with open(cached_features_file, "rb") as handle:
#                 self.examples = pickle.load(handle)
#         else:
#             logger.info("Creating features from dataset file at %s", directory)

#             self.examples = []
#             for iteml, row in df.iterrows():
#             	print(iteml, row)
#                 conv = construct_conv(row, tokenizer)
#                 self.examples.append(conv)

#             logger.info("Saving features into cached file %s", cached_features_file)
#             with open(cached_features_file, "wb") as handle:
#                 pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):
#         return torch.tensor(self.examples[item], dtype=torch.long)
      
# def set_seed():
#     random.seed(1)
#     np.random.seed(1)
#     torch.manual_seed(1)
#     #if args.n_gpu > 0:
#     #    torch.cuda.manual_seed_all(args.seed)