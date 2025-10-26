import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

tokenizer_path = "sapienzanlp/Minerva-7B-instruct-v1.0"
model_path = "./models/DIETA_allsynth.pt"
GENERATE_MAX_LENGTH = 512
print(tokenizer_path)
print(model_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model_trf = TransformerWrapper(
    num_tokens=51200,
    max_seq_len=1024,
    use_abs_pos_emb=False,
    attn_layers=Decoder(
        dim=2048,
        depth=6,
        heads=32,
        pre_norm=False,
        residual_attn=True,
        ff_relu_squared=True,
        attn_add_zero_kv=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        rotary_pos_emb=True,
        rotary_emb_dim=64,
        rotary_xpos_scale_base=32768,
        rotary_xpos=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
    )
)
model_trf = torch.compile(model_trf)
auto_model = AutoregressiveWrapper(model_trf)
#auto_model.cuda()
auto_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print("MODEL LOADED")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("our model has " + str(count_parameters(auto_model)) + " trainable parameters")
auto_model.eval()

sentences = ["ENG: this is a great project! IT:", "IT: Tutto a posto? ENG:"]
for sentence in tqdm(sentences, disable=True):
    input_ids_attnmask = tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = input_ids_attnmask['input_ids']
    sample = auto_model.generate(input_ids, GENERATE_MAX_LENGTH, temperature=0.)
    translation_ = tokenizer.batch_decode(sample)
    translation = ' '.join(translation_).strip()
    print(translation)
