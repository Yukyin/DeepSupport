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

- We provide some comparative examples: see **[`EXAMPLE.md`](EXAMPLE.md)**. 


## Disclaimer ⚠️
- DeepSupport Coach is not professional advice, diagnosis, or consulting (medical, legal, financial, etc.). 

- It provides structured thinking support and action planning. Please make your own decisions and consult qualified professionals when needed.


## Research and Citation 📚
If you use DeepSupport Coach in a paper, report, thesis, or study, please cite this repository:

```bibtex
@software{deepsupport_coach_2026,
  author  = {Yuyan Chen},
  title   = {DeepSupport Coach: A structured thinking companion for clarifying problems and generating next-step plans},
  year    = {2026},
  version = {oss},
  url     = {\url{https://github.com/Yukyin/DeepSupport/DeepSupport_Coach}
}
```


## License 📜
- Noncommercial use (including academic and research use) is governed by `LICENSE` (PolyForm Noncommercial 1.0.0).
- Commercial use requires a separate agreement — see `COMMERCIAL_LICENSE.md`.
- Copyright notice is provided in `NOTICE`.

📨 Commercial inquiries: yolandachen0313@gmail.com
