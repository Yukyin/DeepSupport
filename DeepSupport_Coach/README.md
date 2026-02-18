# DeepSupport Coach 🧑‍🏫
🔮 DeepSupport Coach is a problem-clarification and action-planning companion.  
✍️ Write your blockage, then get clarity, options, and next steps.


![Overview](ds-coach.png)



## What it does ✨
DeepSupport Coach helps you move from a vague problem to a concrete next step:

- 🔎 Reflect and clarify what you’re truly stuck on.  
- 🧩 Break it down into key variables and options.  
- ✅ Generate a practical checklist you can execute immediately.  




## Quickstart 🚀

### 1. Dependencies
```bash
conda create -n deepsupport-coach python=3.10 -y
conda activate deepsupport-coach
cd /path/to/deepsupport-coach
pip install -r requirements.txt
```

### 2. Run with Gradio
```bash
python deepsupport_coach_demo.py \
  --icl_file /path/to/icl_synthetic.json \
  --host 0.0.0.0 --port 7860
```
> In-context learning data is recommended. You can also create your own ICL file by following the format of `icl_synthetic.json`.



## Dataset and Examples 📊
- This OSS release ships with no historical dataset by default. 

- The original internal version relied on private datasets which cannot be open-sourced at this time. For a trial of the original version, please contact the author.

- We provide some comparative examples: see **[`EXAMPLES.md`](EXAMPLES.md)**. 

