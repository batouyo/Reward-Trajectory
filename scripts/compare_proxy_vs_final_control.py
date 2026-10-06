#!/usr/bin/env python3
"""Aggregate paired proxy/final results and render an A/B grid."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
import torch
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--proxy-summary",type=Path,required=True);a.add_argument("--final-summary",type=Path,required=True)
 a.add_argument("--output",type=Path,required=True);x=a.parse_args()
 ss={"proxy":read(x.proxy_summary),"final":read(x.final_summary)}
 out={"source":ss["proxy"]["source"],"prompt":ss["proxy"]["prompt"],"seed":ss["proxy"]["seed"],"steps":ss["proxy"]["steps"],"modes":{},"strength_results":[]}
 for mode,s in ss.items():
  rr=sorted(s["strength_results"],key=lambda z:z["requested_strength"]);pp=[float(z["optimized_final_progress"]) for z in rr]
  gaps={f"gap_{int(a['requested_strength']*100):02d}_{int(b['requested_strength']*100):02d}":float(b["optimized_final_progress"])-float(a["optimized_final_progress"]) for a,b in zip(rr,rr[1:])}
  out["modes"][mode]={"final_progress_is_ordered":all(b>a for a,b in zip(pp,pp[1:])),"level_gaps":gaps,"dynamic_range":max(pp)-min(pp) if pp else 0.0,"probable_failure":s.get("probable_failure")}
  for z in rr:
   out["strength_results"].append({"mode":mode,"strength":z["requested_strength"],"initial_progress":z["initial_proxy_progress"],"optimized_proxy_progress":z["optimized_proxy_progress"],"optimized_final_progress":z["optimized_final_progress"],"final_target_error":z["final_target_error"],"drift":z["optimized_final_drift"],"residual_native_ratio":z["residual_native_ratio"],"residual_rms_per_step":[v["residual_velocity_rms"] for v in z["velocity_diagnostics"]["steps"]],"max_residual_native_rms_ratio":z["velocity_diagnostics"]["max_residual_native_rms_ratio"]})
 out["residual_comparison"]=[]
 for level in sorted({z["strength"] for z in out["strength_results"]}):
  q={m:next(z for z in out["strength_results"] if z["mode"]==m and z["strength"]==level) for m in ("proxy","final")}
  out["residual_comparison"].append({"strength":level,"proxy_ratio":q["proxy"]["residual_native_ratio"],"final_ratio":q["final"]["residual_native_ratio"],"final_residual_larger_than_proxy":q["final"]["residual_native_ratio"]>1.25*q["proxy"]["residual_native_ratio"],"hard_override":q["final"]["residual_native_ratio"]>3.0})
 gd=x.output.parent
 out["gradient_sanity"]={m:read(gd/f"gradient_sanity_{m}.json") for m in ("proxy","final")}
 grads={m:torch.load(gd/f"gradient_sanity_{m}_gradient.pt",map_location="cpu",weights_only=True).float().flatten() for m in ("proxy","final")}
 out["proxy_final_gradient_comparison"]={"proxy_final_gradient_cosine":float(torch.nn.functional.cosine_similarity(grads["proxy"][None],grads["final"][None]).item()),"proxy_gradient_norm":float(grads["proxy"].norm()),"final_gradient_norm":float(grads["final"].norm()),"final_to_proxy_norm_ratio":float(grads["final"].norm()/grads["proxy"].norm().clamp_min(1e-12))}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
 pd,fd=x.proxy_summary.parent,x.final_summary.parent
 files=[(pd/"source.png","source"),(pd/"controlled_s_0.25.png","proxy .25"),(pd/"controlled_s_0.50.png","proxy .50"),(pd/"controlled_s_0.75.png","proxy .75"),(pd/"native_full.png","native full"),(fd/"controlled_s_0.25.png","final .25"),(fd/"controlled_s_0.50.png","final .50"),(fd/"controlled_s_0.75.png","final .75")]
 ims=[(Image.open(f).convert("RGB"),label) for f,label in files];w=max(i.width for i,_ in ims);h=max(i.height for i,_ in ims)
 grid=Image.new("RGB",(w*len(ims),h+30),"white");d=ImageDraw.Draw(grid)
 for n,(im,label) in enumerate(ims):grid.paste(im.resize((w,h)),(n*w,30));d.text((n*w+8,8),label,fill="black")
 gp=x.output.with_name("proxy_vs_final_grid.png");grid.save(gp);out["comparison_grid"]=str(gp.resolve());x.output.write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
