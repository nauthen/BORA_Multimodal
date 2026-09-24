import pathlib,torch,numpy as np,json
from torch.utils.data import DataLoader
from config import TrainConfig
from dataset import load_splits
from dataset.multimodal_dataset import MultimodalFishDataset,multimodal_collate_fn
from models import MultimodalDeepFusionModel
from tasks.trainer import _statistics_from_outputs
from utils.ordinal import bora_loss
from utils.metrics import save_bora_predictions,save_bora_gate_summary,save_confusion_outputs,save_metrics_csv
root=pathlib.Path('/marimo/Multimodal_Deep_Fusion'); out=root/'outputs/PANNS_Cnn6_SwinTiny_bora_fusion/holdout'
cfg=TrainConfig.from_json(out/'config_snapshot.json'); splits=load_splits(cfg.dataset,mode='holdout')
ds=MultimodalFishDataset(splits['test'],'test',cfg.audio_features.sample_rate,cfg.video_features.image_size,True,True,16,num_frames=1)
dl=DataLoader(ds,batch_size=256,shuffle=False,num_workers=8,collate_fn=multimodal_collate_fn,pin_memory=True,persistent_workers=True)
model=MultimodalDeepFusionModel(cfg).cuda().eval(); ckpt=torch.load(out/'checkpoint/multimodal_best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ckpt['model_state_dict'],strict=True);model.set_epoch(int(ckpt['epoch']))
y=[];probs=[];keys=[];rank=[];ar=[];vr=[];ag=[];vg=[];loss_sum=0
with torch.inference_mode():
 for b in dl:
  target=b['target'].cuda(non_blocking=True);o=model(b['waveform'].cuda(non_blocking=True),b['video_form'].cuda(non_blocking=True));loss,_=bora_loss(o,target,cfg.fusion.bora.aux_loss_weight,cfg.fusion.bora.reliability_loss_weight,cfg.fusion.bora.categorical_loss_weight,cfg.fusion.bora.motion_loss_weight);loss_sum+=float(loss)*len(target);y.extend(target.cpu().tolist());probs.extend(torch.softmax(o['clipwise_output'],1).cpu().tolist());keys.extend(b['sample_key']);rank.extend(o['rank_probabilities'].cpu().tolist());ar.extend(o['audio_reliability'].cpu().tolist());vr.extend(o['video_reliability'].cpu().tolist());ag.extend(o['audio_gate_weights'].cpu().tolist());vg.extend(o['video_gate_weights'].cpu().tolist())
st=_statistics_from_outputs(y,probs,4); y_np=np.asarray(y); rank=np.asarray(rank);ar=np.asarray(ar);vr=np.asarray(vr);ag=np.asarray(ag);vg=np.asarray(vg)
result={'best_epoch':float(ckpt['epoch']),'best_val_score':float(ckpt['val_metrics']['accuracy']),'test_loss':loss_sum/len(y),'test_accuracy':float(st['accuracy']),'test_mAP':float(np.mean(st['average_precision'])),'test_precision_macro':float(st['prec_macro']),'test_recall_macro':float(st['rec_macro']),'test_f1_macro':float(st['f1_macro']),'test_precision_weighted':float(st['prec_weighted']),'test_recall_weighted':float(st['rec_weighted']),'test_f1_weighted':float(st['f1_weighted']),'test_rank_mae':float(st['rank_mae']),'test_qwk':float(st['qwk']),'test_within_one_accuracy':float(st['within_one_accuracy']),'test_severe_error_rate':float(st['severe_error_rate'])}
save_metrics_csv(result,out/'result.csv');save_confusion_outputs(y,st['y_pred'].tolist(),out);save_bora_predictions(keys,y_np,st['y_pred'],rank,ar,vr,ag,vg,out/'predictions.csv');save_bora_gate_summary(y_np,ar,vr,ag,vg,out/'gate_summary.csv');(out/'finalized_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
