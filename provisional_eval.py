import json, pathlib, torch, numpy as np
from torch.utils.data import DataLoader
from config import TrainConfig
from dataset import load_splits
from dataset.multimodal_dataset import MultimodalFishDataset, multimodal_collate_fn
from models import MultimodalDeepFusionModel
from tasks.trainer import _statistics_from_outputs
root=pathlib.Path('/marimo/Multimodal_Deep_Fusion')
out=root/'outputs/PANNS_Cnn6_SwinTiny_bora_fusion/holdout'
cfg=TrainConfig.from_json(out/'config_snapshot.json')
splits=load_splits(cfg.dataset,mode='holdout')
dataset=MultimodalFishDataset(splits['test'],'test',cfg.audio_features.sample_rate,cfg.video_features.image_size,True,True,16,num_frames=1)
loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=8,collate_fn=multimodal_collate_fn,pin_memory=True,persistent_workers=True)
model=MultimodalDeepFusionModel(cfg).cuda().eval()
ckpt=torch.load(out/'checkpoint/provisional_epoch43.pt',map_location='cuda',weights_only=False)
model.load_state_dict(ckpt['model_state_dict'],strict=True); model.set_epoch(int(ckpt['epoch']))
y=[]; probs=[]
with torch.inference_mode():
 for batch in loader:
  o=model(batch['waveform'].cuda(non_blocking=True),batch['video_form'].cuda(non_blocking=True))
  y.extend(batch['target'].numpy().astype(int).tolist())
  probs.extend(torch.softmax(o['clipwise_output'],1).cpu().numpy().tolist())
stats=_statistics_from_outputs(y,probs,4)
result={'checkpoint_epoch':int(ckpt['epoch']),'test_accuracy':float(stats['accuracy']),'test_mAP':float(np.mean(stats['average_precision'])),'test_f1_macro':float(stats['f1_macro']),'test_rank_mae':float(stats['rank_mae']),'test_qwk':float(stats['qwk']),'test_within_one_accuracy':float(stats['within_one_accuracy']),'test_severe_error_rate':float(stats['severe_error_rate']),'confusion_matrix':stats['confu_matrix'].tolist()}
(out/'provisional_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
