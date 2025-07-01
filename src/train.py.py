import sys
import torch
##torch._dynamo.config.force_nn_module_property_static_shapes = False
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit =128
import random
import numpy
def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)
import os
from typing import List, Dict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, IterableDataset, Features, Sequence, Value
from torch.utils.data import DataLoader
from tqdm import tqdm
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import transformers
import numpy as np
import bitsandbytes as bnb
import time
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
import GPUtil
from torch.distributed.optim import ZeroRedundancyOptimizer
from datetime import timedelta

def print_gpu_utilization():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"  Load: {gpu.load*100:.1f}%")
        print(f"  Memory Free: {gpu.memoryFree} MB")
        print(f"  Memory Used: {gpu.memoryUsed} MB")
        print(f"  Total Memory: {gpu.memoryTotal} MB\n")
import subprocess

def get_nvidia_smi():
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

import torch.distributed as dist

def setup(rank, world_size):   
    print("SETUP")
    print(rank) 
    print(world_size) 
    os.environ['MASTER_ADDR'] = sys.argv[1] #'localhost'
    os.environ['MASTER_PORT'] = '6000'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))

from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()

tokenizer_path = "sapienzanlp/Minerva-7B-instruct-v1.0"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.eos_token = "</s>"
tokenizer.add_eos_token = True
#print(tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,  # Replace with your tokenizer instance
    mlm=False  # Set to True for Masked Language Model (MLM)
)

def custom_collate_fn(batch):
    # First collate the batch as usual
    collated = data_collator(batch)
    # Return only the input_ids key
    return {"input_ids": collated["input_ids"]}

def run(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    batch_size = 21 #6 layers
    #batch_size = 6 #24 layers
    print("RANK: ")
    print(rank)
    print("\n")
    if rank ==0:
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-000.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data0"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-004.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data4"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-008.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data8"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-012.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data12"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-016.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data16"  
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-020.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data20"    
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/combined_last2M.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data21"            
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/def/formati/final/ea_filterLLM.enit.iten.shuf2.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data22"            
    elif rank==1:
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-001.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data1"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-005.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data5"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-009.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data9"   
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-013.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data13"        
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-017.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data17"        
    elif rank==2:
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-002.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data2"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-006.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data6"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-010.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data10"        
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-014.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data14"        
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-018.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data18"                
    elif rank==3:
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-003.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data3"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-007.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data7"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-011.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data11"        
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-015.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data15"
        train_path = "/leonardo_work/IscrB_modMNMT/ale/data/allfiltered.en-it.it-en.shuf-019.txt.gz"
        cache_dir= "/leonardo_work/IscrB_modMNMT/ale/cache/data19"        
        
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    print(train_path)
    print(cache_dir)
    SAVE_EVERY=25000
    PRINT_EVERY=100
    nb_epoch = 1
    PATH = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_7IT_synthEAep2_"
    #tokenizer_path = "sapienzanlp/Minerva-7B-instruct-v1.0"
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #tokenizer.eos_token = "</s>"
    #tokenizer.add_eos_token = True
    print(tokenizer)

    # prepare the dataloader
    train_dataset, tot_row_batch = prepare(rank, world_size, tokenizer, batch_size, train_path, cache_dir)

    dataloader_t = DataLoader(
        train_dataset,  # Replace with your tokenized and grouped dataset
        batch_size=batch_size,  # Adjust batch size as needed
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        multiprocessing_context="spawn",
    )

    #torch.cuda.set_device(rank)
    # instantiate the model(it's your own model) and move it to the right device
    #model = Model().to(rank)
    model_trf = TransformerWrapper(
        num_tokens = 51200,
        max_seq_len = 1024,
        use_abs_pos_emb = False,   # set this to False
        attn_layers = Decoder(
            dim = 2048,
            depth = 6, #24, #6, ## 24, #12, #6,
            heads = 32,
            #ff_no_bias = True,  # set this to True
            pre_norm = False,#False, #
            residual_attn = True,  # add residual attention
            ff_relu_squared = True,
            attn_add_zero_kv = True, # False, # True,
            attn_dropout = 0.0,    # dropout post-attention
            ff_dropout = 0.0,       # feedforward dropout
            rotary_pos_emb=True,  # turns on rotary positional embeddings
            rotary_emb_dim = 64,
            rotary_xpos_scale_base=32768,
            rotary_xpos = True,   # modified rotary to extrapolate well beyond length at which it was trained
            attn_qk_norm = True,
            attn_qk_norm_dim_scale = True, # set this to True, in addition to `attn_qk_norm = True`
            ##add_value_residual=True,
            ##learned_value_residual_mix = True,
            #attn_kv_heads = 16, # say you want 4 query heads to attend to 1 key / value head
            ##num_residual_streams = 4, # 8 dynamic hyper connection residual streams
            ##integrate_layers = True,
            #use_adaptive_layernorm = True,
            #use_adaptive_layerscale = True,
            #residual_fn_kwargs = dict(
            #    tanh = tanh
            #)
            #resi_dual = True,               # set this to True
            #resi_dual_scale = 0.1,           # in appendix, they said on fp16 the prenorm residual is prone to overflow. they claim by scaling it at each layer by a factor, it would prevent the overflow, and keep results the same (as layernorms are invariant to scaling of the input)
        )
    )
    #print(model_trf)
    model_trf = torch.compile(model_trf)
    auto_model = AutoregressiveWrapper(model_trf)
    auto_model.to(rank)
    auto_model.train()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("our language model has "+str(count_parameters(auto_model))+" trainable parameters")
 
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_1_rank3_50000.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_2IT_1_rank3_50000.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_3IT_1_rank3_50000.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_4IT_1_rank3_50000.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_5IT_1_rank3_50000.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_6IT_1_rank0.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_7IT_1_rank0.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_7IT_synthEA_1_rank0.pt"
    print(PATHMODEL)
    auto_model.load_state_dict(torch.load(PATHMODEL, map_location=lambda storage, loc: storage))
    print("MODEL LOADED")

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model    
    auto_model = DDP(auto_model, device_ids=[rank], output_device=rank)#, find_unused_parameters=True)

    #optimizer = Your_Optimizer()

    #LEARNING_RATE = 2e-4
    LEARNING_RATE = 8e-5
    LEARNING_RATE = 8e-6
    LEARNING_RATE = 5e-6
    #optimizer = bnb.optim.AdamW8bit(auto_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),eps=1e-8,weight_decay=1e-5)#1e-1)#, percentile_clipping=5)
    #optimizer = bnb.optim.Lion8bit(auto_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=1e-5)#1e-1)#, percentile_clipping=5)

    optimizer = ZeroRedundancyOptimizer(auto_model.parameters(), optimizer_class=bnb.optim.Lion8bit, lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=1e-5)
    """
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_1_rank3_50000.optmizer.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_2IT_1_rank3_50000.optmizer.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_3IT_1_rank3_50000.optmizer.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_4IT_1_rank3_50000.optmizer.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_5IT_1_rank3_50000.optmizer.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_6IT_1_rank0.optmizer.pt"
    print(PATHMODEL)
    optimizer.load_state_dict(torch.load(PATHMODEL, map_location=lambda storage, loc: storage))
    print("OPTIM LOADED")
    """
    print(optimizer)
    t_total = tot_row_batch #300000 #28836851 #14418425 # 28836851 #500
    percentage=10 #5
    warmup_steps = int((percentage / 100) * t_total)
    print(warmup_steps)
    print(t_total)
    #warmup_steps*=5
    #t_total*=5
    #print(warmup_steps)
    #print(t_total)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    """
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_1_rank3_50000.scheduler.pt"
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_2IT_1_rank3_50000.scheduler.pt"   
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_3IT_1_rank3_50000.scheduler.pt"    
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_4IT_1_rank3_50000.scheduler.pt"  
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_5IT_1_rank3_50000.scheduler.pt"   
    PATHMODEL = "/leonardo_work/IscrB_modMNMT/ale/model/model6layers_6IT_1_rank0.scheduler.pt"    
    print(PATHMODEL)
    scheduler.load_state_dict(torch.load(PATHMODEL, map_location=lambda storage, loc: storage))
    print("SCHEDULER LOADED")
    """

    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    # Start the timer
    start_time = time.time()
    clip=1.0
    i = 0
    devi = 'cuda:'+str(rank)
    print(devi)
    print(get_nvidia_smi())
    #loss_fn = Your_Loss()    
    for epoch in range(nb_epoch):
        for item in tqdm(dataloader_t, disable=True):
            #print(item)
            i+=1
            #sys.exit(1)
            input_ids = item['input_ids'].to(devi, non_blocking=True) 
            #print(input_ids.shape)        
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = auto_model(input_ids)
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(auto_model.parameters(), clip)

            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            scheduler.step()

            if i % PRINT_EVERY == 0 and rank == 0:
                tqdm.write(f"Step: {i}")
                tqdm.write(f"Loss: {loss.item()}")
                #If print_gpu_utilization() prints output, you may need to modify it to return a string.
                ######tqdm.write(print_gpu_utilization())
                #tqdm.write(get_nvidia_smi())
                sys.stdout.flush()

            #print(i)
            #if i % PRINT_EVERY ==0:
            #    print(i)
            #    print(loss)
            #    print_gpu_utilization()
            #    #print(scheduler.get_last_lr())
            #print("loss")
            #i+=1
            if i % SAVE_EVERY == 0:
                ep = epoch+1
                print("save every "+str(SAVE_EVERY))        
                end_time = time.time()
                execution_time = end_time - start_time
                #print("TEMPO")
                print(execution_time)
                start_time = time.time()
                #optimizer.consolidate_state_dict()
                #if rank ==0:
                #    torch.save(auto_model.module.state_dict(), PATH+str(ep)+".pt")
                #    torch.save(optimizer.state_dict(), PATH+str(ep)+".optmizer.pt")
                #    torch.save(scheduler.state_dict(), PATH+str(ep)+".scheduler.pt")
                #sys.exit(1)
                sys.stdout.flush()
                dist.barrier()  # Synchronize all processes
                #for rank in range(dist.get_world_size()):
                optimizer.consolidate_state_dict(to=0)
                if rank ==0:#3:
                    torch.save(auto_model.module.state_dict(), PATH+str(ep)+"_rank"+str(rank)+"_"+str(i)+".pt")
                    torch.save(optimizer.state_dict(), PATH+str(ep)+"_rank"+str(rank)+"_"+str(i)+".optmizer.pt")
                    torch.save(scheduler.state_dict(), PATH+str(ep)+"_rank"+str(rank)+"_"+str(i)+".scheduler.pt")


        print(f"Rank {rank}: starting final checkpoint saving")
        sys.stdout.flush()
        dist.barrier()  # Synchronize all processes
        print(f"Rank {rank}: passed barrier in final checkpoint")
        sys.stdout.flush()
        #for rank in range(dist.get_world_size()):
        #    optimizer.consolidate_state_dict(to=rank)
        torch.cuda.synchronize()
        optimizer.consolidate_state_dict(to=0)
        print(f"Rank {rank}: completed optimizer consolidation")
        sys.stdout.flush()
        ep = epoch+1
        if rank ==0:#3:
            torch.save(auto_model.module.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".pt")
            torch.save(optimizer.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".optmizer.pt")
            torch.save(scheduler.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".scheduler.pt")
       
        #ep = epoch+1
        #if (ep%1) ==0:
        #    print("save EPOCH")
        #    #dist.barrier()
        #    #torch.save(auto_model.module.state_dict(), PATH+str(ep)+".pt")
        #    #torch.save(optimizer.state_dict(), PATH+str(ep)+".optmizer.pt")
        #    #torch.save(scheduler.state_dict(), PATH+str(ep)+".scheduler.pt")
        #    for rank in dist.get_world_size():
        #        optimizer.consolidate_state_dict(rank)
        #    torch.save(auto_model.module.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".pt")
        #    torch.save(optimizer.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".optmizer.pt")
        #    torch.save(scheduler.state_dict(), PATH+str(ep)+"_rank"+str(rank)+".scheduler.pt")
        #    #if rank ==0:
        #    #    torch.save(auto_model.module.state_dict(), PATH+str(ep)+".pt")
        #    #    torch.save(optimizer.state_dict(), PATH+str(ep)+".optmizer.pt")
        #    #    torch.save(scheduler.state_dict(), PATH+str(ep)+".scheduler.pt")
    dist.barrier()
    #cleanup()

import torch.multiprocessing as mp
if __name__ == '__main__':
    world_size = 1
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size
    )


def prepare(rank, world_size, tokenizer, batch_size, train_path, cache_dir):
    #dataset = Your_Dataset()
    #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    #dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, 
    #                        drop_last=False, shuffle=False, sampler=sampler)



    sequence_length = 1024
    nb_epoch = 1

    dataset = load_dataset("text", data_files=train_path, split="train", cache_dir=cache_dir)  # , streaming=True)


    def group_texts_ending_with_zero(examples: Dict[str, List[np.ndarray]], sequence_length: int) -> Dict[str, List[np.ndarray]]:

        concatenated_examples = list(examples.values())[0]

        chunks = []            # will hold all chunks (each chunk is a list of length `sequence_length`)
        current_chunk = []     # tokens accumulated in the current chunk
        current_length = 0     # how many tokens in current_chunk so far

        for arr in concatenated_examples:
            arr_len = len(arr)

            # If the array itself is too big to fit in an empty chunk, skip it
            if arr_len > sequence_length:
                print(f"Skipping array of length {arr_len} (exceeds {sequence_length})")
                continue

            # If it doesn't fit in the remaining space, finalize (pad) the current chunk
            if current_length + arr_len > sequence_length:
                # pad current chunk with -1 up to `sequence_length`
                while current_length < sequence_length:
                    current_chunk.append(tokenizer.unk_token_id)
                    current_length += 1

                #chunks.append(current_chunk)
                chunks.append(np.array(current_chunk, dtype=np.int64))
                # start a new chunk
                current_chunk = []
                current_length = 0

            # Now arr fits in the current chunk
            current_chunk.extend(arr)
            current_length += arr_len

        # After processing all arrays, if there's anything leftover in current_chunk, pad and store
        if current_length > 0 and current_length < sequence_length:
            while current_length < sequence_length:
                current_chunk.append(tokenizer.unk_token_id)
                current_length += 1
            #chunks.append(current_chunk)
            chunks.append(np.array(current_chunk, dtype=np.int64))

        result = {'input_ids': chunks}
        return result


    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i: i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        #print(texts)
        tokenized_batch = tokenizer.batch_encode_plus(texts['text'], return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        #return group_texts(tokenized_batch)
        return group_texts_ending_with_zero(tokenized_batch, sequence_length)


    train_dataset = dataset.map(
        _tokenize_and_group_texts,
        #input_columns=text_column_name,
        remove_columns='text', #raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length)}), #+ 1
        batched=True,
        num_proc=1, #dataset_processing_num_proc_per_process,
        load_from_cache_file=True, #not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length}",
    )
    print(train_dataset)
    num_rows = train_dataset.num_rows
    print(num_rows)
    tot_row_batch = num_rows//batch_size
    print(tot_row_batch)

    #def collate_fn_cuda(batch):
    #    # Collate the batch using your existing data_collator
    #    collated_batch = data_collator(batch)
    #    # Move each tensor in the batch to CUDA
    #    collated_batch = {key: value.to('cuda', non_blocking=True) 
    #                      for key, value in collated_batch.items()}
    #    return collated_batch

    return train_dataset, tot_row_batch





print("done")
