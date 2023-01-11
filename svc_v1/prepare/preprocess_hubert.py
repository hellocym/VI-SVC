import os
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import joblib


from fairseq import checkpoint_utils
from huggingface_hub import snapshot_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/TencentGameMate/chinese_speech_pretrain
snapshot_download(repo_id="TencentGameMate/chinese-hubert-base", 
ignore_regex=["*.gitattributes", "*.md", "*.bin"],
cache_dir='.')

model_path = "./models--TencentGameMate--chinese-hubert-base/snapshots/fce0375452b1dd6c080ac3248d423d4d037bc831/chinese-hubert-base-fairseq-ckpt.pt"
kmeans_model_path = "./chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl"  # layer 9


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
print("load model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
model = model.half()
model.eval()

# K-means model
print(f"Loading K-means model from {kmeans_model_path} ...")
kmeans_model = joblib.load(open(kmeans_model_path, "rb"))
kmeans_model.verbose = False


wavPath = "data/waves"
outPath = "data/phone"

for spks in os.listdir(wavPath):
    if os.path.isdir(f"./{wavPath}/{spks}"):
        if not os.path.exists(f"./{outPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}")
        for file in os.listdir(f"./{wavPath}/{spks}"):
            if file.endswith(".wav"):
                wav_path = f"./{wavPath}/{spks}/{file}"
                out_path = f"./{outPath}/{spks}/{file[:-4]}.npy"

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": feats.half().to(device),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9,  # layer 9
                }

                with torch.no_grad():
                    feats, _ = model.extract_features(**inputs)

                feats = feats.squeeze(0).float().cpu().numpy()
                pred = kmeans_model.predict(feats)
                pred = np.repeat(pred, 2) # 20ms -> 10ms

                np.save(out_path, pred, allow_pickle=False)
