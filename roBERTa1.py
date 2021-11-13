# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 02:49:19 2021

@author: hp
"""


from tokenizers import ByteLevelBPETokenizer
import os
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments




paths = r'C:\Users\hp\deep learning\Transformers-for-Natural-Language-Processing-main\Chapter03\kant.txt'
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, 
                special_tokens=[
                                "<pad>",
                                "<s>",
                                "</s>",
                                "<unk>",
                                "<mask>",
                                ]       
                )


token_dir = r'C:\Users\hp\deep learning\KantaiBERT'
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model(token_dir)

tokenizer = ByteLevelBPETokenizer(
                                    r'C:\Users\hp\deep learning\KantaiBERT\vocab.json',
                                    r'C:\Users\hp\deep learning\KantaiBERT\merges.txt',
                                    )
print(tokenizer.encode("The Critique of Pure Reason.").tokens)
print(tokenizer.token_to_id("</s>"))
print(tokenizer.token_to_id("<s>"))
tokenizer._tokenizer.post_processor = BertProcessing(
                                                        ("</s>", tokenizer.token_to_id("</s>")),
                                                        ("<s>", tokenizer.token_to_id("<s>")),
                                                        )
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("The Critique of Pure Reason.").tokens)
config = RobertaConfig(
                        vocab_size=52_000,
                        max_position_embeddings=514,
                        num_attention_heads=12,
                        num_hidden_layers=6,
                        type_vocab_size=1,
                        )
tokenizer = RobertaTokenizer.from_pretrained(r'C:\Users\hp\deep learning\KantaiBERT', max_length=512)
model = RobertaForMaskedLM(config=config)
print(model)
print(model.num_parameters())

dataset = LineByLineTextDataset(
                                tokenizer=tokenizer,
                                file_path= r'C:\Users\hp\deep learning\Transformers-for-Natural-Language-Processing-main\Chapter03\kant.txt',
                                block_size=128,
                                )

#@title Step 11: Defining a Data Collator

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
                                    output_dir=r'C:\Users\hp\deep learning\KantaiBERT',
                                    overwrite_output_dir=True,
                                    num_train_epochs=1,
                                    per_device_train_batch_size=64,
                                    save_steps=10_000,
                                    save_total_limit=2,
                                    )
trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset,
                    )
trainer.train()
trainer.save_model(r'C:\Users\hp\deep learning\KantaiBERT')



